import re
import hashlib
import tempfile
import subprocess
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
    ):
        self.c_path = Path(c_path)
        self.extra_sources = [Path(p) for p in (extra_sources or [])]
        self.include_dirs = list(include_dirs or [])
        self.compiler = compiler
        self.compile_args = compile_args or ["-O3", "-shared", "-fPIC"]
        self.auto_wrap = auto_wrap
        self.auto_cdef = auto_cdef

        self._ffi = None
        self._lib = None
        self._sig = None
        self._so_path = None

        # will be filled after parsing
        self._func_specs: dict[str, FuncSpec] = {}

        # if cdef not provided, generate it from C file
        if cdef is None and auto_cdef:
            cdef = self._generate_cdef_from_source()
        self.cdef = cdef

        self.ensure_compiled()

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
        scalar_args = [a for a in fspec.args if a.is_scalar]
        length_args = [a for a in fspec.args if a.is_length_param]

        if len(length_args) > 1:
            raise NotImplementedError(
                f"{fspec.name}: multiple length params not supported yet"
            )

        # validate array types
        for a in array_args:
            _numpy_dtype_for_base_type(a.base_type)

        def wrapper(*args):
            self.ensure_compiled()
            lib = self.lib
            ffi = self.ffi
            cfun = getattr(lib, fspec.name)

            out_array_specs = [
                a for a in array_args if a.is_array_out and a.name.lower().startswith("out")
            ]
            in_array_specs = [a for a in array_args if a not in out_array_specs]

            if len(args) != len(in_array_specs) + len(scalar_args):
                raise TypeError(
                    f"{fspec.name} expects {len(in_array_specs)} array args and "
                    f"{len(scalar_args)} scalar args, got {len(args)}"
                )

            py_array_vals = args[: len(in_array_specs)]
            py_scalar_vals = args[len(in_array_specs) :]

            # prepare arrays
            c_array_ptrs = []
            lengths = []
            output_arrays: list[np.ndarray] = []

            # user-provided (input) arrays
            for a, val in zip(in_array_specs, py_array_vals):
                base_dtype = _numpy_dtype_for_base_type(a.base_type)
                arr = np.ascontiguousarray(val, dtype=base_dtype)
                lengths.append(arr.size)
                ctype = f"{'const ' if a.is_const else ''}{a.base_type} *"
                ptr = ffi.cast(ctype, ffi.from_buffer(arr))
                c_array_ptrs.append((a, arr, ptr))

            # auto-created output arrays
            for a in out_array_specs:
                base_dtype = _numpy_dtype_for_base_type(a.base_type)
                ref_arr = None
                if a.name.lower().startswith("out_"):
                    ref_name = a.name[4:]
                    for (aa, arr, _ptr) in c_array_ptrs:
                        if aa.name == ref_name:
                            ref_arr = arr
                            break
                if ref_arr is not None:
                    arr = np.empty_like(ref_arr, dtype=base_dtype)
                else:
                    target_len = lengths[0] if lengths else None
                    if target_len is None:
                        raise ValueError(
                            f"{fspec.name}: cannot determine length for output array {a.name}"
                        )
                    arr = np.empty(target_len, dtype=base_dtype)
                lengths.append(arr.size)
                ctype = f"{a.base_type} *"
                ptr = ffi.cast(ctype, ffi.from_buffer(arr))
                c_array_ptrs.append((a, arr, ptr))
                output_arrays.append(arr)

            if lengths:
                n0 = lengths[0]
                for ln in lengths[1:]:
                    if ln != n0:
                        raise ValueError("Array arguments must have same length")
                inferred_len = n0
            else:
                inferred_len = None

            scalar_vals_iter = iter(py_scalar_vals)
            c_args = []

            for a in fspec.args:
                if a.is_array_in or a.is_array_out:
                    for (aa, arr, ptr) in c_array_ptrs:
                        if aa is a:
                            c_args.append(ptr)
                            break
                    else:
                        raise RuntimeError("Internal error mapping array arg")
                elif a.is_length_param:
                    if inferred_len is None:
                        raise ValueError(
                            f"{fspec.name}: cannot infer length for {a.name}"
                        )
                    c_args.append(inferred_len)
                elif a.is_scalar:
                    try:
                        sv = next(scalar_vals_iter)
                    except StopIteration:
                        raise TypeError("Not enough scalar arguments")
                    c_args.append(sv)
                else:
                    raise RuntimeError("Arg classification inconsistent")

            res = cfun(*c_args)

            if output_arrays:
                outputs = output_arrays[0] if len(output_arrays) == 1 else tuple(output_arrays)
                if fspec.return_ctype.strip() == "void":
                    return outputs
                return outputs, res

            if fspec.return_ctype.strip() == "void":
                return None
            else:
                return res

        wrapper.__name__ = fspec.name
        doc_lines = [f"Auto-wrapped C function `{fspec.name}`."]
        if fspec.doc:
            doc_lines.append(fspec.doc)
        wrapper.__doc__ = "\n".join(doc_lines)
        return wrapper
