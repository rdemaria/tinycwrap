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


@dataclass
class StructField:
    name: str
    raw_ctype: str
    base_type: str
    is_pointer: bool
    is_const: bool
    array_len: int | None = None


@dataclass
class StructSpec:
    name: str
    fields: list[StructField]


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
        *c_sources,
        include_dirs=None,
        compiler="gcc",
        compile_args=None,
        reload=True,
    ):
        """
        Parameters
        ----------
        c_sources : str or Path
            One or more C source files to compile. The first is treated as the primary file
            for cdef generation and doc extraction; others are linked in.
        include_dirs : list[str | Path], optional
            Additional include directories for the compiler.
        compiler : str, default "gcc"
            C compiler executable to invoke.
        compile_args : list[str], optional
            Extra compiler flags. Defaults to ["-O3", "-shared", "-fPIC"].
        reload : bool, default True
            If True and running inside IPython, register a pre-run-cell hook to auto-recompile
            when the C source changes.
        """
        if not c_sources:
            raise ValueError("At least one C source path is required")
        self._c_path = Path(c_sources[0])
        self._extra_sources = [Path(p) for p in c_sources[1:]]
        self._compile_options = {
            "include_dirs": list(include_dirs or []),
            "compiler": compiler,
            "compile_args": compile_args
            or ["-O3", "-shared", "-fPIC", "-march=native", "-mtune=native"],
        }

        self._ffi = None
        self._lib = None
        self._sig = None
        self._so_path = None
        self._cdef: str | None = None

        # will be filled after parsing
        self._func_specs: dict[str, FuncSpec] = {}
        self._struct_specs: dict[str, StructSpec] = {}
        self._struct_dtypes: dict[str, np.dtype] = {}
        self._struct_classes: dict[str, type] = {}
        self._ipython_hook = None
        self._cdef: str | None = None

        self._ensure_compiled()
        if reload:
            try:
                self._register_ipython_autoreload()
            except Exception:
                pass

    # ---------- build & reload ---------------------------------------------

    def _compute_sig(self):
        h = hashlib.sha1()
        all_paths = [self._c_path] + self._extra_sources
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

        so_name = f"cmodule_{self._c_path.stem}_{sig}.so"
        so_path = build_dir / so_name

        cmd = [self._compile_options["compiler"], *self._compile_options["compile_args"]]

        include_dirs = self._compile_options["include_dirs"] + [np.get_include()]
        for inc in include_dirs:
            cmd.extend(["-I", str(inc)])

        # regenerate cdef from source
        self._cdef = self._generate_cdef_from_source(verbose=self._cdef is None)

        sources = [str(self._c_path), *(str(p) for p in self._extra_sources)]
        cmd.extend(["-o", str(so_path), *sources])

        print(f"[CModule] Compiling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        ffi = FFI()
        ffi.cdef(self._cdef)
        lib = ffi.dlopen(str(so_path))

        self._ffi = ffi
        self._lib = lib
        self._sig = sig
        self._so_path = so_path

        print(f"[CModule] Loaded {so_path}")

        # Re-parse cdef and attach docs from C source, then create wrappers
        self._func_specs = self._parse_cdef(self._cdef)
        self._struct_specs = self._parse_structs_from_cdef(self._cdef)
        self._attach_docs_from_source()
        self._create_struct_classes()
        self._create_wrappers()

    def _ensure_compiled(self):
        if self._needs_recompile():
            self._compile_and_load()

    # ---------- IPython auto-reload hook ------------------------------------

    def _register_ipython_autoreload(self):
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
                self._ensure_compiled()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[CModule] Auto-compile failed: {exc}")

        ip.events.register("pre_run_cell", _hook)
        self._ipython_hook = _hook

    def _unregister_ipython_autoreload(self):
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

    def _generate_cdef_from_source(self, verbose: bool = True) -> str:
        """
        Parse the C file and auto-generate a minimal cdef string.

        We look for *definitions* of non-static functions of the form:

            [static] <ret> name(<args>)
            /* optional doc */
            {

        and turn them into:
            <ret> name(<args>);
        """
        src = self._c_path.read_text(encoding="utf8")

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
        struct_defs = []

        struct_re = re.compile(
            r"typedef\s+struct\s*{(?P<body>[^}]*)}\s*(?P<name>\w+)\s*;",
            re.DOTALL,
        )
        for m in struct_re.finditer(src_wo_pp):
            struct_text = m.group(0)
            struct_defs.append(struct_text.strip())

        for m in func_def_re.finditer(src_wo_pp):
            if m.group("prefix") and "static" in m.group("prefix"):
                # ignore static functions, not exported
                continue
            ret = " ".join(_strip_restrict_keywords(m.group("ret")).split())
            if ret.strip().startswith(("for", "while", "if")):
                continue
            name = m.group("name")
            args = " ".join(_strip_restrict_keywords(m.group("args")).split())
            proto = f"{ret} {name}({args});"
            prototypes.append(proto)

        if not prototypes and not struct_defs:
            raise RuntimeError(f"No functions found in {self._c_path} to generate cdef")

        cdef_parts = []
        if struct_defs:
            cdef_parts.extend(struct_defs)
        if prototypes:
            cdef_parts.extend(prototypes)
        cdef_lines = []
        for line in "\n".join(cdef_parts).splitlines():
            if re.match(r"\s*(for|while|if)\b", line):
                continue
            cdef_lines.append(line)
        cdef = "\n".join(cdef_lines)
        if verbose:
            print("[CModule] Auto-generated cdef:\n" + cdef)
        return cdef

    # ---------- 2) parse cdef -> FuncSpec/ArgSpec ---------------------------

    def _parse_cdef(self, cdef: str) -> dict[str, FuncSpec]:
        text = cdef
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text = _strip_restrict_keywords(text)

        # remove struct definitions for function parsing
        text = re.sub(r"typedef\s+struct\s*{[^}]*}\s*\w+\s*;", "", text, flags=re.DOTALL)

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

    def _parse_structs_from_cdef(self, cdef: str) -> dict[str, StructSpec]:
        structs: dict[str, StructSpec] = {}
        text = _strip_restrict_keywords(cdef)
        struct_re = re.compile(
            r"typedef\s+struct\s*{(?P<body>[^}]*)}\s*(?P<name>[A-Za-z_]\w*)\s*;",
            re.DOTALL | re.MULTILINE,
        )
        for m in struct_re.finditer(text):
            body = m.group("body")
            name = m.group("name")
            fields: list[StructField] = []
            body_clean = re.sub(r"\bstruct\b", "", body)
            for line in body_clean.split(";"):
                line = line.strip()
                if not line:
                    continue
                # handle possible trailing comments
                line = re.sub(r"/\*.*?\*/", "", line).strip()
                m_field = re.match(r"(.+?)\s+([A-Za-z_]\w*)(\s*\[(\d+)\])?$", line)
                if not m_field:
                    continue
                raw_ctype = m_field.group(1).strip()
                fname = m_field.group(2)
                array_len = int(m_field.group(4)) if m_field.group(4) else None
                is_pointer = "*" in raw_ctype
                is_const = "const" in raw_ctype
                base = _base_type_from_ctype(raw_ctype)
                try:
                    _numpy_dtype_for_base_type(base)
                except TypeError:
                    continue
                fields.append(
                    StructField(
                        name=fname,
                        raw_ctype=raw_ctype,
                        base_type=base,
                        is_pointer=is_pointer,
                        is_const=is_const,
                        array_len=array_len,
                    )
                )
            if fields:
                structs[name] = StructSpec(name=name, fields=fields)
        return structs

    # ---------- 3) attach docstrings from C comments ------------------------

    def _attach_docs_from_source(self):
        try:
            src = self._c_path.read_text(encoding="utf8")
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

    def _create_struct_classes(self):
        for sname, sspec in self._struct_specs.items():
            try:
                dtype_fields = []
                for f in sspec.fields:
                    base_dtype = _numpy_dtype_for_base_type(f.base_type)
                    if f.array_len:
                        dtype_fields.append((f.name, (base_dtype, (f.array_len,))))
                    else:
                        dtype_fields.append((f.name, base_dtype))
                dtype = np.dtype(dtype_fields)
            except TypeError:
                continue

            def make_struct_class(spec, dtype):
                slots = ("_data",)

                def __init__(self, **kwargs):
                    data = np.zeros((), dtype=dtype)
                    for k in dtype.names:
                        if k in kwargs:
                            data[k] = kwargs[k]
                    object.__setattr__(self, "_data", data)

                def __repr__(self):
                    parts = ", ".join(f"{name}={self._data[name].item()!r}" for name in dtype.names)
                    return f"{spec.name}({parts})"

                namespace = {
                    "__slots__": slots,
                    "__init__": __init__,
                    "__repr__": __repr__,
                    "__doc__": f"Python wrapper for C struct {spec.name}. Fields: {', '.join(dtype.names)}.",
                    "dtype": dtype,
                    "zeros": staticmethod(lambda n, _dtype=dtype: np.zeros(n, dtype=_dtype)),
                }

                for fname in dtype.names:
                    field_info = dtype.fields[fname][0]
                    is_scalar = field_info.shape == ()
                    def getter(self, fname=fname, is_scalar=is_scalar):
                        val = self._data[fname]
                        return val.item() if is_scalar else val

                    def setter(self, value, fname=fname, field_info=field_info, is_scalar=is_scalar):
                        if is_scalar:
                            self._data[fname] = value
                        else:
                            arr = np.asarray(value, dtype=field_info.base)
                            if arr.shape != field_info.shape:
                                arr = np.reshape(arr, field_info.shape)
                            np.copyto(self._data[fname], arr)

                    namespace[fname] = property(getter, setter)

                return type(spec.name, (), namespace)

            struct_cls = make_struct_class(sspec, dtype)
            self._struct_dtypes[sname] = dtype
            self._struct_classes[sname] = struct_cls
            setattr(self, sname, struct_cls)

    def _make_wrapper_from_spec(self, fspec: FuncSpec):
        struct_names = set(self._struct_specs.keys())
        struct_ptr_args = [a for a in fspec.args if a.is_pointer and a.base_type in struct_names]
        array_args = [a for a in fspec.args if (a.is_array_in or a.is_array_out) and a not in struct_ptr_args]
        length_args = [a for a in fspec.args if a.is_length_param]

        if len(length_args) > 1:
            raise NotImplementedError(f"{fspec.name}: multiple length params not supported yet")

        # validate array types
        for a in array_args:
            _numpy_dtype_for_base_type(a.base_type)

        src = self._build_wrapper_source(fspec)
        namespace = {
            "_self": self,
            "np": np,
            "_numpy_dtype_for_base_type": _numpy_dtype_for_base_type,
            "_struct_classes": self._struct_classes,
            "_struct_dtypes": self._struct_dtypes,
        }
        filename = f"<cmodule:{self._c_path.name}:{fspec.name}>"
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
            wrapper.__c_source__ = self._c_path.read_text(encoding="utf8")
        except OSError:
            wrapper.__c_source__ = None
        return wrapper

    def _build_wrapper_source(self, fspec: FuncSpec) -> str:
        """
        Build the Python source string for a wrapper with an explicit signature.
        Keeping this separate allows inspection/debugging of the generated code.
        """
        try:
            c_source_text = self._c_path.read_text(encoding="utf8")
        except OSError:
            c_source_text = None

        struct_names = set(self._struct_specs.keys())
        params: list[str] = []
        for a in fspec.args:
            if a.is_array_out and a.name.lower().startswith("out"):
                params.append(f"{a.name}=None")
            elif a.is_length_param:
                params.append(f"{a.name}=None")
            elif a.is_pointer and a.base_type in struct_names and not a.is_const and a.name.lower().startswith("out"):
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
            if a.is_pointer and a.base_type in struct_names:
                dtype_expr = f"_struct_dtypes['{a.base_type}']"
                cls_expr = f"_struct_classes['{a.base_type}']"
                if not a.is_const and a.name.lower().startswith("out"):
                    lines += [
                        f"    if {a.name} is None:",
                        f"        arr_{a.name} = np.zeros((), dtype={dtype_expr})",
                        f"    elif isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    ]
                    output_vars.append(f"arr_{a.name}")
                else:
                    lines += [
                        f"    if isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    ]
                lines += [
                    f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                ]
                call_args.append(f"ptr_{a.name}")
                continue

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
                        f"            arr_{a.name} = np.empty_like(ref_arr, dtype=base_dtype)",
                        "        else:",
                    ]
                if length_name:
                    lines += [
                        f"            arr_{a.name} = np.empty(int({length_name}), dtype=base_dtype)",
                    ]
                else:
                    lines += [
                        f"            raise ValueError('{fspec.name}: missing length parameter for output array {a.name}')",
                    ]
                lines += [
                    "    else:",
                    f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype=base_dtype)",
                    f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                ]
                call_args.append(f"ptr_{a.name}")
                output_vars.append(f"arr_{a.name}")
                array_var_names.append(a.name)
            elif a.is_array_in or a.is_array_out:
                const_prefix = "const " if a.is_const else ""
                dtype_expr = f"np.dtype('{np.dtype(_numpy_dtype_for_base_type(a.base_type)).name}')"
                lines += [
                    f"    arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    f"    ptr_{a.name} = _self._ffi.cast('{const_prefix}{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                ]
                call_args.append(f"ptr_{a.name}")
                array_var_names.append(a.name)
            elif a.is_pointer and a.base_type in struct_names:
                dtype_expr = f"_struct_dtypes['{a.base_type}']"
                cls_expr = f"_struct_classes['{a.base_type}']"
                if not a.is_const and a.name.lower().startswith("out"):
                    lines += [
                        f"    if {a.name} is None:",
                        f"        arr_{a.name} = np.zeros((), dtype={dtype_expr})",
                        f"    elif isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    ]
                else:
                    lines += [
                        f"    if isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                    ]
                lines += [
                    f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                ]
                call_args.append(f"ptr_{a.name}")
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

        if any((a.is_array_in or a.is_array_out) and a.base_type not in struct_names for a in fspec.args):
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
