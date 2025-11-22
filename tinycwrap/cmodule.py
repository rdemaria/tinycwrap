import re
import hashlib
import tempfile
import subprocess
import linecache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cffi import FFI


# ---------------- helpers --------------------------------------------------


def _strip_restrict_keywords(text: str) -> str:
    """Remove C restrict qualifiers (including compiler-specific variants)."""
    return re.sub(r"\b(__restrict__|__restrict|restrict)\b", "", text)


def _base_type_from_ctype(ctype: str) -> str:
    """Normalize base C type (strip const, *, etc.)."""
    ctype = _strip_restrict_keywords(ctype)
    ctype = ctype.replace("const", "").replace("volatile", "")
    ctype = ctype.replace("*", "").strip()
    return " ".join(ctype.split())


def _numpy_dtype_for_base_type(base: str):
    """Map base C type -> numpy dtype."""
    if base == "double":
        return np.float64
    if base == "float":
        return np.float32
    if base in ("int", "signed int"):
        return np.int32
    if base in ("long long", "long long int", "signed long long"):
        return np.int64
    if base in ("unsigned int", "unsigned"):
        return np.uint32
    # Extend here if you need more types
    raise TypeError(f"Unsupported C base type for NumPy mapping: {base!r}")


def _is_length_name(name: str) -> bool:
    name = name.lower()
    if name in ("n", "len", "length", "size"):
        return True
    return name.startswith("len_") or name.startswith("n_") or name.startswith("size_")


@dataclass
class ArgSpec:
    name: str
    raw_ctype: str
    base_type: str
    is_pointer: bool
    is_const: bool
    is_length_param: bool = False
    is_array_in: bool = False
    is_array_out: bool = False
    is_scalar: bool = False


@dataclass
class FuncSpec:
    name: str
    return_ctype: str
    args: list   # list[ArgSpec]
    doc: str | None = None


# ---------------- main class ----------------------------------------------


class CModule:
    """
    Compile & hot-reload a C file into a shared library and:

      * auto-generate CFFI cdef from function definitions
      * auto-generate NumPy-friendly Python wrappers

    Conventions:

      * We export all **non-static** functions found in the C file.
      * Arguments:
          - `const double *x`  -> input NumPy array
          - `double *x`        -> in-place / output NumPy array
          - `int len_x`, `int n`, `int size_x` -> length parameter (hidden)
          - other scalars (int/double/float/long long) -> Python scalars
      * A block comment immediately after the function header is used
        as the Python docstring, e.g.:

            double dot(const double *x, const double *y, int len_x)
            /* Return dot product between x and y */
            {
                ...
            }
    """

    def __init__(
        self,
        c_path,
        cdef: str | None = None,
        extra_sources=None,
        include_dirs=None,
        compiler="gcc",
        compile_args=None,
        auto_wrap=True,
        auto_cdef=True,
        reload=True,
    ):
        """
        Parameters
        ----------
        c_path : str or Path
            Path to the primary C source file to compile.
        cdef : str, optional
            Explicit CFFI cdef string. If None and auto_cdef is True, it is generated from the source.
        extra_sources : list[str | Path], optional
            Additional C source files to compile and link.
        include_dirs : list[str | Path], optional
            Additional include directories for the compiler.
        compiler : str, default "gcc"
            C compiler executable to invoke.
        compile_args : list[str], optional
            Extra compiler flags. Defaults to ["-O3", "-shared", "-fPIC"].
        auto_wrap : bool, default True
            Whether to generate Python wrappers for exported C functions.
        auto_cdef : bool, default True
            Whether to auto-generate the cdef from the C source when cdef is None.
        reload : bool, default True
            If True and running inside IPython, register a pre-run-cell hook to auto-recompile
            when the C source changes.
        """
        self.c_path = Path(c_path)
        self.extra_sources = [Path(p) for p in (extra_sources or [])]
        self.include_dirs = list(include_dirs or [])
        self.compiler = compiler
        self.compile_args = compile_args or ["-O3", "-shared", "-fPIC"]
        self.auto_wrap = auto_wrap
        self.auto_cdef = auto_cdef
        self._auto_cdef_from_source = cdef is None and auto_cdef

        self._ffi = None
        self._lib = None
        self._sig = None
        self._so_path = None

        # will be filled after parsing
        self._func_specs: dict[str, FuncSpec] = {}
        self._ipython_hook = None

        # if cdef not provided, generate it from C file
        if self._auto_cdef_from_source:
            cdef = self._generate_cdef_from_source()
        self.cdef = cdef

        self.ensure_compiled()
        if reload:
            try:
                self.register_ipython_autoreload()
            except Exception:
                pass

    # ---------- build & reload ---------------------------------------------

    def _compute_sig(self):
        h = hashlib.sha1()
        all_paths = [self.c_path] + self.extra_sources
        for p in all_paths:
            st = p.stat()
            h.update(str(p.resolve()).encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
        return h.hexdigest()[:16]

    def _needs_recompile(self):
        if self._sig is None:
            return True
        return self._compute_sig() != self._sig

    def _compile_and_load(self):
        sig = self._compute_sig()
        build_dir = Path(tempfile.gettempdir()) / "cmodule_build"
        build_dir.mkdir(parents=True, exist_ok=True)

        so_name = f"cmodule_{self.c_path.stem}_{sig}.so"
        so_path = build_dir / so_name

        cmd = [self.compiler, *self.compile_args]

        include_dirs = self.include_dirs + [np.get_include()]
        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])

        # regenerate cdef if we auto-generate from source
        if self._auto_cdef_from_source:
            self.cdef = self._generate_cdef_from_source()

        sources = [str(self.c_path), *(str(p) for p in self.extra_sources)]
        cmd.extend(["-o", str(so_path), *sources])

        print(f"[CModule] Compiling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        ffi = FFI()
        ffi.cdef(self.cdef)
        lib = ffi.dlopen(str(so_path))

        self._ffi = ffi
        self._lib = lib
        self._sig = sig
        self._so_path = so_path

        print(f"[CModule] Loaded {so_path}")

        # Re-parse cdef and attach docs from C source, then create wrappers
        if self.auto_wrap:
            self._func_specs = self._parse_cdef(self.cdef)
            self._attach_docs_from_source()
            self._create_wrappers()

    def ensure_compiled(self):
        if self._needs_recompile():
            self._compile_and_load()

    # ---------- properties ---------------------------------------------------

    @property
    def ffi(self):
        self.ensure_compiled()
        return self._ffi

    @property
    def lib(self):
        self.ensure_compiled()
        return self._lib

    # ---------- IPython auto-reload hook ------------------------------------

    def register_ipython_autoreload(self):
        """Register a pre-run-cell hook in IPython to auto-recompile if sources changed."""
        try:
            from IPython import get_ipython  # type: ignore
        except ImportError as exc:
            raise RuntimeError("IPython is required for auto-reload") from exc
        ip = get_ipython()
        if ip is None:
            raise RuntimeError("No active IPython session")
        if self._ipython_hook is not None:
            return

        def _hook(*_args, **_kwargs):
            try:
                self.ensure_compiled()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[CModule] Auto-compile failed: {exc}")

        ip.events.register("pre_run_cell", _hook)
        self._ipython_hook = _hook

    def unregister_ipython_autoreload(self):
        """Remove the IPython pre-run-cell hook if registered."""
        hook = self._ipython_hook
        if hook is None:
            return
        try:
            from IPython import get_ipython  # type: ignore
        except ImportError:
            self._ipython_hook = None
            return
        ip = get_ipython()
        if ip is not None:
            try:
                ip.events.unregister("pre_run_cell", hook)
            except Exception:
                pass
        self._ipython_hook = None

    # ---------- 1) auto-generate cdef from C source -------------------------

    def _generate_cdef_from_source(self) -> str:
        """
        Parse the C file and auto-generate a minimal cdef string.

        We look for *definitions* of non-static functions of the form:

            [static] <ret> name(<args>)
            /* optional doc */
            {

        and turn them into:
            <ret> name(<args>);
        """
        src = self.c_path.read_text(encoding="utf8")

        # Remove preprocessor lines to simplify
        src_wo_pp = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)

        # regex for function definitions
        func_def_re = re.compile(
            r"""
            (?P<prefix>static\s+)?            # optional 'static'
            (?P<ret>[^{}();]+?)              # return type
            \s+
            (?P<name>\w+)\s*                 # function name
            \(
                (?P<args>[^)]*)              # arguments (no nested parentheses)
            \)
            \s*
            (?:/\*.*?\*/\s*)?                # optional trailing comment
            \{                               # function body begins
            """,
            re.VERBOSE | re.DOTALL,
        )

        prototypes = []

        for m in func_def_re.finditer(src_wo_pp):
            if m.group("prefix") and "static" in m.group("prefix"):
                # ignore static functions, not exported
                continue
            ret = " ".join(_strip_restrict_keywords(m.group("ret")).split())
            name = m.group("name")
            args = " ".join(_strip_restrict_keywords(m.group("args")).split())
            proto = f"{ret} {name}({args});"
            prototypes.append(proto)

        if not prototypes:
            raise RuntimeError(f"No functions found in {self.c_path} to generate cdef")

        cdef = "\n".join(prototypes)
        print("[CModule] Auto-generated cdef:\n" + cdef)
        return cdef

    # ---------- 2) parse cdef -> FuncSpec/ArgSpec ---------------------------

    def _parse_cdef(self, cdef: str) -> dict[str, FuncSpec]:
        text = cdef
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text = _strip_restrict_keywords(text)

        decls = []
        buff = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            buff.append(line)
            if ";" in line:
                decl = " ".join(buff)
                decls.append(decl)
                buff = []

        func_re = re.compile(r"(.+?)\s+(\w+)\s*\((.*?)\)\s*;")

        funcs: dict[str, FuncSpec] = {}

        for decl in decls:
            m = func_re.match(decl)
            if not m:
                continue
            ret_ctype, fname, arglist = m.groups()
            ret_ctype = ret_ctype.strip()
            arglist = arglist.strip()

            if arglist == "void" or arglist == "":
                argspecs = []
            else:
                argspecs = []

            for raw_arg in re.split(r"\s*,\s*", arglist):
                raw_arg = raw_arg.strip()
                if not raw_arg:
                    continue

                # Match: [type stuff possibly with *] [name]
                # Handles: "const double *x", "double* x", "double * x", etc.
                m_arg = re.match(r"(.+?)\s*([A-Za-z_]\w*)$", raw_arg)
                if not m_arg:
                    # couldn't parse this arg, skip or raise
                    # raise ValueError(f"Cannot parse argument: {raw_arg!r}")
                    continue

                ctype = m_arg.group(1).strip()
                name = m_arg.group(2)

                is_pointer = "*" in ctype
                is_const = "const" in ctype
                base = _base_type_from_ctype(ctype)

                aspec = ArgSpec(
                    name=name,
                    raw_ctype=ctype,
                    base_type=base,
                    is_pointer=is_pointer,
                    is_const=is_const,
                )
                argspecs.append(aspec)


            # classify
            for a in argspecs:
                if (not a.is_pointer) and _is_length_name(a.name) and a.base_type in (
                    "int",
                    "unsigned int",
                    "unsigned",
                ):
                    a.is_length_param = True

            for a in argspecs:
                if a.is_pointer:
                    if a.is_const:
                        a.is_array_in = True
                    else:
                        a.is_array_out = True
                else:
                    if not a.is_length_param:
                        a.is_scalar = True

            funcs[fname] = FuncSpec(name=fname, return_ctype=ret_ctype, args=argspecs)

        return funcs

    # ---------- 3) attach docstrings from C comments ------------------------

    def _attach_docs_from_source(self):
        try:
            src = self.c_path.read_text(encoding="utf8")
        except OSError:
            return

        for fname, fspec in self._func_specs.items():
            # we look for: name ( ... ) /* ... */
            pattern = rf"{re.escape(fname)}\s*\([^{{;]*\)\s*/\*(.*?)\*/"
            m = re.search(pattern, src, flags=re.DOTALL)
            if not m:
                continue
            doc = m.group(1).strip()
            doc = re.sub(r"\s+\n", "\n", doc)
            doc = re.sub(r"\n\s+", "\n", doc)
            fspec.doc = doc

    # ---------- 4) create NumPy wrappers ------------------------------------

    def _create_wrappers(self):
        for fname, fspec in self._func_specs.items():
            if hasattr(self, fname):
                continue
            try:
                wrapper = self._make_wrapper_from_spec(fspec)
            except NotImplementedError:
                # too complex for our heuristics, skip
                continue
            setattr(self, fname, wrapper)

    def _make_wrapper_from_spec(self, fspec: FuncSpec):
        array_args = [a for a in fspec.args if a.is_array_in or a.is_array_out]
        length_args = [a for a in fspec.args if a.is_length_param]

        if array_args and not length_args:
            raise NotImplementedError(
                f"{fspec.name}: array arguments require an explicit length parameter"
            )
        if len(length_args) > 1:
            raise NotImplementedError(
                f"{fspec.name}: multiple length params not supported yet"
            )

        # validate array types
        for a in array_args:
            _numpy_dtype_for_base_type(a.base_type)

        src = self._build_wrapper_source(fspec)
        namespace = {
            "_self": self,
            "np": np,
            "_numpy_dtype_for_base_type": _numpy_dtype_for_base_type,
        }
        filename = f"<cmodule:{self.c_path.name}:{fspec.name}>"
        linecache.cache[filename] = (
            len(src),
            None,
            [line + "\n" for line in src.splitlines()],
            filename,
        )
        code = compile(src, filename, "exec")
        exec(code, namespace)
        wrapper = namespace[fspec.name]
        wrapper.__source__ = src
        try:
            wrapper.__c_source__ = self.c_path.read_text(encoding="utf8")
        except OSError:
            wrapper.__c_source__ = None
        return wrapper

    def _build_wrapper_source(self, fspec: FuncSpec) -> str:
        """
        Build the Python source string for a wrapper with an explicit signature.
        Keeping this separate allows inspection/debugging of the generated code.
        """
        try:
            c_source_text = self.c_path.read_text(encoding="utf8")
        except OSError:
            c_source_text = None

        params: list[str] = []
        for a in fspec.args:
            if a.is_array_out and a.name.lower().startswith("out"):
                params.append(f"{a.name}=None")
            elif a.is_length_param:
                params.append(f"{a.name}=None")
            else:
                params.append(a.name)

        signature = ", ".join(params)
        arg_docs: list[str] = []
        for a in fspec.args:
            role = "scalar"
            if a.is_array_in:
                role = "array in"
            elif a.is_array_out:
                role = "array out"
            elif a.is_length_param:
                role = "length"
            ctype = a.raw_ctype.strip()
            extra = []
            if a.is_array_out and a.name.lower().startswith("out"):
                extra.append("auto if None")
            if a.is_length_param:
                extra.append("auto from array length")
            extra_txt = f" [{' '.join(extra)}]" if extra else ""
            arg_docs.append(f"{a.name} : {ctype} ({role}){extra_txt}")

        doc_lines = [f"Auto-wrapped C function `{fspec.name}`."]
        if fspec.doc:
            doc_lines.append(fspec.doc)
        if arg_docs:
            doc_lines.extend(
                [
                    "",
                    "Parameters",
                    "----------",
                    *arg_docs,
                ]
            )
        if c_source_text:
            doc_lines.extend(
                [
                    "",
                    "C source",
                    "--------",
                    *(line for line in c_source_text.splitlines()),
                ]
            )

        lines = [
            f"def {fspec.name}({signature}):",
            '    """' + ("\n    ".join(doc_lines)) + '"""',
            f"    cfun = getattr(_self._lib, '{fspec.name}')",
        ]

        output_vars: list[str] = []
        call_args: list[str] = []
        array_var_names: list[str] = []
        length_args = [a for a in fspec.args if a.is_length_param]
        length_name = length_args[0].name if length_args else None

        for a in fspec.args:
            if a.is_array_out and a.name.lower().startswith("out"):
                dtype_expr = f"np.dtype('{np.dtype(_numpy_dtype_for_base_type(a.base_type)).name}')"
                ref_name = a.name[4:] if a.name.lower().startswith("out_") else None
                lines += [
                    f"    base_dtype = {dtype_expr}",
                    f"    if {a.name} is None:",
                ]
                if ref_name:
                    lines += [
                        f"        ref_arr = locals().get('arr_{ref_name}', None)",
                        "        if ref_arr is not None:",
                        "            arr = np.empty_like(ref_arr, dtype=base_dtype)",
                        "        else:",
                    ]
                if length_name:
                    lines += [
                        f"            arr = np.empty(int({length_name}), dtype=base_dtype)",
                    ]
                else:
                    lines += [
                        f"            raise ValueError('{fspec.name}: missing length parameter for output array {a.name}')",
                    ]
                lines += [
                    "    else:",
                    f"        arr = np.ascontiguousarray({a.name}, dtype=base_dtype)",
                    f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr))",
                    f"    arr_{a.name} = arr",
                ]
                call_args.append(f"ptr_{a.name}")
                output_vars.append(f"arr_{a.name}")
                array_var_names.append(a.name)
            elif a.is_array_in or a.is_array_out:
                const_prefix = "const " if a.is_const else ""
                dtype_expr = f"np.dtype('{np.dtype(_numpy_dtype_for_base_type(a.base_type)).name}')"
                lines += [
                    f"    arr = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    f"    ptr_{a.name} = _self._ffi.cast('{const_prefix}{a.base_type} *', _self._ffi.from_buffer(arr))",
                    f"    arr_{a.name} = arr",
                ]
                call_args.append(f"ptr_{a.name}")
                array_var_names.append(a.name)
            elif a.is_length_param:
                # infer from naming convention if not provided
                base_arr = None
                if a.name.startswith("len_"):
                    base_arr = a.name[4:]
                elif a.name.startswith("n_"):
                    base_arr = a.name[2:]
                elif a.name.startswith("size_"):
                    base_arr = a.name[5:]
                target_arr = base_arr if base_arr in array_var_names else (array_var_names[0] if array_var_names else None)
                if target_arr:
                    lines += [
                        f"    if {a.name} is None:",
                        f"        {a.name} = len(arr_{target_arr})",
                        f"    {a.name} = int({a.name})",
                    ]
                else:
                    lines += [
                        f"    if {a.name} is None:",
                        f"        raise ValueError('{fspec.name}: cannot infer length for {a.name}')",
                        f"    {a.name} = int({a.name})",
                    ]
                call_args.append(a.name)
            elif a.is_scalar:
                lines += [f"    {a.name} = {a.name}"]
                call_args.append(a.name)

        if any(a.is_array_in or a.is_array_out for a in fspec.args):
            if not length_name:
                lines += [
                    f"    raise ValueError('{fspec.name}: array arguments require a length parameter')"
                ]

        ret_type = fspec.return_ctype.strip()
        call_expr = f"cfun({', '.join(call_args)})"
        if ret_type == "void":
            lines += [
                f"    {call_expr}",
            ]
        else:
            lines += [
                f"    res = {call_expr}",
            ]

        if output_vars:
            tuple_expr = output_vars[0] if len(output_vars) == 1 else f"({', '.join(output_vars)})"
            if ret_type == "void":
                lines += [
                    "    outputs = " + tuple_expr,
                    "    return outputs",
                ]
            else:
                lines += [
                    "    outputs = " + tuple_expr,
                    "    return outputs, res",
                ]
        else:
            if ret_type == "void":
                lines.append("    return None")
            else:
                lines.append("    return res")

        return "\n".join(lines)
