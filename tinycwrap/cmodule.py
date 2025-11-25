import re
import hashlib
import tempfile
import subprocess
import linecache
from pathlib import Path

import numpy as np
from cffi import FFI

from .parsing import (
    FuncSpec,
    StructSpec,
    numpy_dtype_for_base_type,
    parse_functions_from_cdef,
    parse_structs_from_cdef,
    strip_restrict_keywords,
)


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

        cmd = [
            self._compile_options["compiler"],
            *self._compile_options["compile_args"],
        ]

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
        self._func_specs = parse_functions_from_cdef(self._cdef)
        self._struct_specs = parse_structs_from_cdef(self._cdef)
        self._attach_docs_from_source()
        self._mark_length_params_from_contracts()
        self._create_struct_classes()
        self._create_wrappers()

    def _ensure_compiled(self):
        if self._needs_recompile():
            # Reset wrapper and struct state before recompiling
            for fname in list(self._func_specs.keys()):
                if hasattr(self, fname):
                    try:
                        delattr(self, fname)
                    except Exception:
                        pass
            for sname in list(self._struct_classes.keys()):
                if hasattr(self, sname):
                    try:
                        delattr(self, sname)
                    except Exception:
                        pass
            self._func_specs.clear()
            self._struct_specs.clear()
            self._struct_dtypes.clear()
            self._struct_classes.clear()
            self._ffi = None
            self._lib = None
            self._so_path = None
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
        src_parts: list[str] = []
        for path in [self._c_path, *self._extra_sources]:
            hdr = path.with_suffix(".h")
            if hdr.exists():
                try:
                    src_parts.append(hdr.read_text(encoding="utf8"))
                except OSError:
                    pass
            try:
                src_parts.append(path.read_text(encoding="utf8"))
            except OSError:
                continue
        src = "\n".join(src_parts)

        # Remove preprocessor lines to simplify
        src_wo_pp = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
        # Strip common compiler-specific noise that confuses the regex parser
        src_wo_pp = re.sub(r"__attribute__\s*\(\([^)]*\)\)", "", src_wo_pp)
        src_wo_pp = re.sub(r"__declspec\s*\([^)]+\)", "", src_wo_pp)
        src_wo_pp = re.sub(r"__asm__\s*\([^)]*\)", "", src_wo_pp)

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
            r"typedef\s+struct\s+(?:\w+\s*)?{(?P<body>[^}]*)}\s*(?P<name>\w+)\s*;",
            re.DOTALL,
        )
        for m in struct_re.finditer(src_wo_pp):
            struct_text = m.group(0)
            struct_defs.append(struct_text.strip())

        proto_set = set()
        for m in func_def_re.finditer(src_wo_pp):
            if m.group("prefix") and "static" in m.group("prefix"):
                # ignore static functions, not exported
                continue
            bad_keywords = {
                "for",
                "while",
                "if",
                "else",
                "switch",
                "case",
                "return",
                "do",
            }
            ret = " ".join(strip_restrict_keywords(m.group("ret")).split())
            first_ret_token = ret.strip().split()[0] if ret.strip() else ""
            if first_ret_token in bad_keywords:
                continue
            name = m.group("name")
            if name in bad_keywords:
                continue
            args = " ".join(strip_restrict_keywords(m.group("args")).split())
            proto = f"{ret} {name}({args});"
            if proto not in proto_set:
                prototypes.append(proto)
                proto_set.add(proto)

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
            contracts = []
            for line in doc.splitlines():
                if "Contract:" in line or "Post-Contract:" in line:
                    after = (
                        line.split("Contract:", 1)[1]
                        if "Contract:" in line
                        else line.split("Post-Contract:", 1)[1]
                    )
                    is_post = "Post-Contract" in line
                    for part in after.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        mlen = re.match(r"len\((\w+)\)\s*=\s*(.+)", part)
                        if not mlen:
                            mlen = re.match(r"(\w+)\s*=\s*(.+)", part)
                        if mlen:
                            target = mlen.group(1)
                            expr = mlen.group(2).strip()
                            contracts.append((target, expr, is_post))
            fspec.contracts = contracts or None

    def _mark_length_params_from_contracts(self):
        """Mark length-like parameters based solely on explicit contracts."""
        for fspec in self._func_specs.values():
            referenced: set[str] = set()
            if fspec.contracts:
                for target, expr, _ in fspec.contracts:
                    referenced.add(target)
                    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
                        if tok == "len":
                            continue
                        referenced.add(tok)
            for arg in fspec.args:
                arg.is_length_param = False
            for arg in fspec.args:
                if (
                    arg.base_type
                    in (
                        "int",
                        "unsigned int",
                        "unsigned",
                        "long",
                        "long int",
                        "long long",
                        "long long int",
                        "unsigned long",
                        "unsigned long long",
                        "size_t",
                        "ssize_t",
                    )
                    and (not arg.is_pointer)
                    and arg.name in referenced
                ):
                    arg.is_length_param = True
            for arg in fspec.args:
                if arg.is_length_param:
                    arg.is_scalar = False
                elif (
                    (not arg.is_pointer)
                    and arg.array_len is None
                    and not arg.is_array_in
                    and not arg.is_array_out
                ):
                    arg.is_scalar = True

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
                    base_dtype = numpy_dtype_for_base_type(f.base_type)
                    if f.array_len:
                        dtype_fields.append((f.name, (base_dtype, (f.array_len,))))
                    else:
                        dtype_fields.append((f.name, base_dtype))
                dtype = np.dtype(dtype_fields, align=True)
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
                    parts_list = []
                    for name in dtype.names:
                        val = self._data[name]
                        if val.shape == ():
                            parts_list.append(f"{name}={val.item()!r}")
                        else:
                            parts_list.append(f"{name}={val.tolist()!r}")
                    parts = ", ".join(parts_list)
                    return f"{spec.name}({parts})"

                namespace = {
                    "__slots__": slots,
                    "__init__": __init__,
                    "__repr__": __repr__,
                    "__doc__": f"Python wrapper for C struct {spec.name}. Fields: {', '.join(dtype.names)}.",
                    "dtype": dtype,
                    "zeros": staticmethod(
                        lambda n, _dtype=dtype: np.zeros(n, dtype=_dtype)
                    ),
                }

                for fname in dtype.names:
                    field_info = dtype.fields[fname][0]
                    is_scalar = field_info.shape == ()

                    def getter(self, fname=fname, is_scalar=is_scalar):
                        val = self._data[fname]
                        return val.item() if is_scalar else val

                    def setter(
                        self,
                        value,
                        fname=fname,
                        field_info=field_info,
                        is_scalar=is_scalar,
                    ):
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
        struct_ptr_args = [
            a for a in fspec.args if a.is_pointer and a.base_type in struct_names
        ]
        array_args = [
            a
            for a in fspec.args
            if (a.is_array_in or a.is_array_out) and a not in struct_ptr_args
        ]

        # validate array types (skip structs, handled separately)
        for a in array_args:
            if a.base_type in struct_names:
                continue
            numpy_dtype_for_base_type(a.base_type)

        src = self._build_wrapper_source(fspec)
        namespace = {
            "_self": self,
            "np": np,
            "numpy_dtype_for_base_type": numpy_dtype_for_base_type,
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
        len_params: list[str] = []
        for a in fspec.args:
            if a.is_array_out and a.name.lower().startswith("out"):
                params.append(f"{a.name}=None")
            elif a.is_length_param:
                len_params.append(f"{a.name}=None")
            elif (
                a.is_pointer
                and a.base_type in struct_names
                and not a.is_const
                and a.name.lower().startswith("out")
            ):
                params.append(f"{a.name}=None")
            else:
                params.append(a.name)

        params.extend(len_params)
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
                extra.append("auto from Contract")
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

        contract_map: dict[str, str] = {}
        post_contract_map: dict[str, str] = {}
        if getattr(fspec, "contracts", None):
            for target, expr, is_post in fspec.contracts:
                if is_post:
                    post_contract_map[target] = expr
                else:
                    contract_map[target] = expr

        pointer_scalar_names = {
            a.name
            for a in fspec.args
            if (
                a.is_array_out
                and not a.is_array_in
                and a.array_len is None
                and a.base_type
                in (
                    "int",
                    "unsigned int",
                    "unsigned",
                    "long",
                    "long int",
                    "long long",
                    "long long int",
                    "size_t",
                    "ssize_t",
                )
            )
        }

        def _expr_py(expr: str) -> str:
            expr = expr.strip()
            expr = re.sub(r"len\((\w+)\)", r"len(arr_\1)", expr)
            for name in pointer_scalar_names:
                expr = re.sub(rf"\b{name}\b", f"int(arr_{name}.ravel()[0])", expr)
            return expr

        call_args: list[str] = []
        output_vars: list[str] = []
        output_names: list[str] = []
        pointer_scalar_outputs: list[tuple[str, str]] = []
        struct_scalar_outputs: list[tuple[str, str, str]] = []
        pre_lines: list[str] = []
        length_lines: list[str] = []
        out_lines: list[str] = []
        scalar_lines: list[str] = []

        for a in fspec.args:
            if a.is_array_in:
                const_prefix = "const " if a.is_const else ""
                if a.base_type in struct_names:
                    cls_expr = f"_struct_classes['{a.base_type}']"
                    dtype_expr = f"_struct_dtypes['{a.base_type}']"
                    pre_lines += [
                        f"    if isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                        f"    ptr_{a.name} = _self._ffi.cast('{const_prefix}{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                    ]
                else:
                    dtype_expr = f"np.dtype('{np.dtype(numpy_dtype_for_base_type(a.base_type)).name}')"
                    if (a.array_len is None) and (a.name not in contract_map):
                        pre_lines += [
                            f"    if np.ndim({a.name}) == 0:",
                            f"        arr_{a.name} = np.ascontiguousarray(np.array({a.name}, dtype={dtype_expr}))",
                            "    else:",
                            f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                            f"    ptr_{a.name} = _self._ffi.cast('{const_prefix}{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                        ]
                    else:
                        pre_lines += [
                            f"    arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                            f"    ptr_{a.name} = _self._ffi.cast('{const_prefix}{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                        ]
                call_args.append(f"ptr_{a.name}")
            elif a.is_scalar and not a.is_length_param:
                scalar_lines.append(f"    {a.name} = {a.name}")
                call_args.append(a.name)
            elif a.is_length_param:
                if a.name in contract_map:
                    expr_py = _expr_py(contract_map[a.name])
                    length_lines += [
                        f"    if {a.name} is None:",
                        f"        {a.name} = int({expr_py})",
                        f"    else:",
                        f"        {a.name} = int({a.name})",
                    ]
                else:
                    length_lines += [
                        f"    if {a.name} is None:",
                        f"        raise ValueError('{fspec.name}: length parameter {a.name} requires an explicit Contract')",
                        f"    {a.name} = int({a.name})",
                    ]
                call_args.append(a.name)
            else:
                # defer outputs / pointer handling
                call_args.append(None)  # placeholder

        # second pass for outputs and non-const struct pointers
        arg_call_args: list[str] = call_args.copy()
        for idx, a in enumerate(fspec.args):
            if a.is_array_out and not a.is_array_in:
                if a.base_type in struct_names:
                    dtype_expr = f"_struct_dtypes['{a.base_type}']"
                else:
                    dtype_expr = f"np.dtype('{np.dtype(numpy_dtype_for_base_type(a.base_type)).name}')"
                ref_name = a.name[4:] if a.name.lower().startswith("out_") else None
                out_lines += [
                    f"    base_dtype = {dtype_expr}",
                    f"    if {a.name} is None:",
                ]
                if a.array_len is not None:
                    out_lines += [
                        f"        arr_{a.name} = np.empty({int(a.array_len)}, dtype=base_dtype)"
                    ]
                elif a.name in contract_map:
                    expr_py = _expr_py(contract_map[a.name])
                    out_lines += [
                        f"        arr_{a.name} = np.empty(int({expr_py}), dtype=base_dtype)"
                    ]
                elif a.array_len is None and a.name not in contract_map:
                    out_lines += [
                        f"        arr_{a.name} = np.zeros((), dtype=base_dtype)"
                    ]
                elif ref_name:
                    out_lines += [
                        f"        ref_arr = locals().get('arr_{ref_name}', None)",
                        "        if ref_arr is not None:",
                        f"            arr_{a.name} = np.empty_like(ref_arr, dtype=base_dtype)",
                        "        else:",
                        f"            raise ValueError('{fspec.name}: provide {a.name} or a Contract for its length')",
                    ]
                else:
                    out_lines += [
                        f"        raise ValueError('{fspec.name}: provide {a.name} or a Contract for its length')",
                    ]
                out_lines += [
                    "    else:",
                    f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype=base_dtype)",
                    f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                ]
                arg_call_args[idx] = f"ptr_{a.name}"
                output_names.append(a.name)
                if a.name in pointer_scalar_names:
                    pointer_scalar_outputs.append((a.name, f"arr_{a.name}"))
                else:
                    output_vars.append(f"arr_{a.name}")
                    if (
                        a.base_type in struct_names
                        and a.array_len is None
                        and a.name not in contract_map
                        and ref_name is None
                        and not a.is_array_in
                    ):
                        struct_scalar_outputs.append(
                            (a.name, f"arr_{a.name}", f"_struct_classes['{a.base_type}']")
                        )
            elif a.is_pointer and a.base_type in struct_names:
                dtype_expr = f"_struct_dtypes['{a.base_type}']"
                cls_expr = f"_struct_classes['{a.base_type}']"
                if a.is_const:
                    out_lines += [
                        f"    if isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                        f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                    ]
                else:
                    out_lines += [
                        f"    if {a.name} is None:",
                    ]
                    if a.array_len is not None:
                        out_lines += [
                            f"        arr_{a.name} = np.zeros({int(a.array_len)}, dtype={dtype_expr})"
                        ]
                    elif a.name in contract_map:
                        expr_py = _expr_py(contract_map[a.name])
                        out_lines += [
                            f"        arr_{a.name} = np.zeros(int({expr_py}), dtype={dtype_expr})"
                        ]
                    elif a.array_len is None and a.name not in contract_map:
                        out_lines += [
                            f"        arr_{a.name} = np.zeros((), dtype={dtype_expr})"
                        ]
                    else:
                        out_lines += [
                            f"        raise ValueError('{fspec.name}: provide {a.name} or a Contract for its length')"
                        ]
                    out_lines += [
                        f"    elif isinstance({a.name}, {cls_expr}):",
                        f"        arr_{a.name} = {a.name}._data",
                        "    else:",
                        f"        arr_{a.name} = np.ascontiguousarray({a.name}, dtype={dtype_expr})",
                        f"    ptr_{a.name} = _self._ffi.cast('{a.base_type} *', _self._ffi.from_buffer(arr_{a.name}))",
                    ]
                    output_names.append(a.name)
                    output_vars.append(f"arr_{a.name}")
                arg_call_args[idx] = f"ptr_{a.name}"
            elif a.is_scalar and not a.is_length_param and arg_call_args[idx] is None:
                scalar_lines.append(f"    {a.name} = {a.name}")
                arg_call_args[idx] = a.name

        lines.extend(pre_lines)
        lines.extend(length_lines)
        lines.extend(out_lines)
        lines.extend(scalar_lines)

        ret_type = fspec.return_ctype.strip()
        call_expr = f"cfun({', '.join(arg_call_args)})"
        if ret_type == "void":
            lines.append(f"    {call_expr}")
        else:
            lines.append(f"    res = {call_expr}")

        for out_name, out_var in zip(output_names, output_vars):
            if out_name in post_contract_map:
                expr_py = _expr_py(post_contract_map[out_name])
                lines.append(f"    {out_var} = {out_var}[:int({expr_py})]")

        for name, arr_var in pointer_scalar_outputs:
            lines.append(f"    scalar_{name} = int(np.asarray({arr_var}).ravel()[0])")

        struct_scalar_map = {arr: (name, cls_expr) for name, arr, cls_expr in struct_scalar_outputs}
        for name, arr_var, cls_expr in struct_scalar_outputs:
            lines.append(f"    obj_{name} = {cls_expr}()")
            lines.append(f"    object.__setattr__(obj_{name}, '_data', np.array({arr_var}, copy=True))")
        output_vars_final: list[str] = []
        for ov in output_vars:
            if ov in struct_scalar_map:
                n, _ = struct_scalar_map[ov]
                output_vars_final.append(f"obj_{n}")
            else:
                output_vars_final.append(ov)

        if output_vars_final:
            if ret_type == "void":
                if len(output_vars_final) == 1:
                    ret_expr = output_vars_final[0]
                else:
                    ret_expr = "(" + ", ".join(output_vars_final) + ")"
            else:
                if len(output_vars_final) == 1:
                    ret_expr = f"{output_vars_final[0]}, res"
                else:
                    ret_expr = "(" + ", ".join(output_vars_final) + "), res"
            if pointer_scalar_outputs:
                scalars = [f"scalar_{n}" for n, _ in pointer_scalar_outputs]
                if isinstance(ret_expr, str) and ret_expr.startswith("("):
                    ret_expr = (
                        ret_expr[:-1]
                        + (", " if len(scalars) else "")
                        + ", ".join(scalars)
                        + ")"
                    )
                else:
                    if len(scalars) == 1:
                        ret_expr = (
                            "(" + ret_expr + ", " + scalars[0] + ")"
                            if ret_expr
                            else scalars[0]
                        )
                    else:
                        ret_expr = "(" + ret_expr + ", " + ", ".join(scalars) + ")"
            lines.append(f"    return {ret_expr}")
        else:
            if ret_type == "void":
                lines.append("    return None")
            else:
                ret_expr = "res"
                if pointer_scalar_outputs:
                    scalars = [f"scalar_{n}" for n, _ in pointer_scalar_outputs]
                    if len(scalars) == 1:
                        ret_expr = f"({ret_expr}, {scalars[0]})"
                    else:
                        ret_expr = f"({ret_expr}, " + ", ".join(scalars) + ")"
                lines.append(f"    return {ret_expr}")

        return "\n".join(lines)
