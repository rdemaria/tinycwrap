import re
from dataclasses import dataclass

import numpy as np

try:
    from pycparser import c_parser, c_ast  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    c_parser = None
    c_ast = None

__all__ = [
    "ArgSpec",
    "FuncSpec",
    "StructField",
    "StructSpec",
    "strip_restrict_keywords",
    "base_type_from_ctype",
    "numpy_dtype_for_base_type",
    "is_length_name",
    "parse_functions_from_cdef",
    "parse_structs_from_cdef",
]


def strip_restrict_keywords(text: str) -> str:
    """Remove C restrict qualifiers (including compiler-specific variants)."""
    return re.sub(r"\b(__restrict__|__restrict|restrict)\b", "", text)


def base_type_from_ctype(ctype: str) -> str:
    """Normalize base C type (strip const, *, etc.)."""
    ctype = strip_restrict_keywords(ctype)
    ctype = ctype.replace("const", "").replace("volatile", "")
    ctype = ctype.replace("*", "").strip()
    return " ".join(ctype.split())


def numpy_dtype_for_base_type(base: str):
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
    raise TypeError(f"Unsupported C base type for NumPy mapping: {base!r}")


def is_length_name(name: str) -> bool:
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


def parse_functions_from_cdef(cdef: str) -> dict[str, FuncSpec]:
    if c_parser is not None:
        try:
            return _parse_functions_with_pycparser(cdef)
        except Exception:
            pass  # fallback to regex
    return _parse_functions_regex(cdef)


def parse_structs_from_cdef(cdef: str) -> dict[str, StructSpec]:
    if c_parser is not None:
        try:
            return _parse_structs_with_pycparser(cdef)
        except Exception:
            pass  # fallback to regex
    return _parse_structs_regex(cdef)


# ---------- pycparser helpers -----------------------------------------------


def _ctype_from_decl(decl) -> tuple[str, bool, bool, int | None]:
    """
    Return (ctype_str, is_pointer, is_const, array_len) from a pycparser decl node.
    """
    is_pointer = False
    is_const = False
    array_len = None
    node = decl
    if getattr(decl, "quals", None) and "const" in decl.quals:
        is_const = True
    while True:
        if isinstance(node, c_ast.TypeDecl):
            names = node.type.names if isinstance(node.type, c_ast.IdentifierType) else []
            ctype = " ".join(names)
            if getattr(node, "quals", None) and "const" in node.quals:
                is_const = True
            return ctype, is_pointer, is_const, array_len
        if isinstance(node, c_ast.PtrDecl):
            is_pointer = True
            if node.quals and "const" in node.quals:
                is_const = True
            node = node.type
            continue
        if isinstance(node, c_ast.ArrayDecl):
            dim = node.dim.value if isinstance(node.dim, c_ast.Constant) else None
            array_len = int(dim) if dim is not None else None
            node = node.type
            continue
        break
    return "", is_pointer, is_const, array_len


def _parse_functions_with_pycparser(cdef: str) -> dict[str, FuncSpec]:
    parser = c_parser.CParser()
    wrapped = f"{cdef}\n"
    ast = parser.parse(wrapped)
    funcs: dict[str, FuncSpec] = {}
    for ext in ast.ext:
        if isinstance(ext, c_ast.Decl) and isinstance(ext.type, c_ast.FuncDecl):
            fname = ext.name
            ret_ctype, _, _, _ = _ctype_from_decl(ext.type.type)
            argspecs: list[ArgSpec] = []
            args = ext.type.args
            if args and args.params:
                for param in args.params:
                    if isinstance(param, c_ast.EllipsisParam):
                        continue
                    ctype, is_ptr, is_const, _ = _ctype_from_decl(param.type)
                    base = base_type_from_ctype(ctype)
                    arg = ArgSpec(
                        name=param.name or "",
                        raw_ctype=ctype,
                        base_type=base,
                        is_pointer=is_ptr,
                        is_const=is_const,
                    )
                    argspecs.append(arg)
            for a in argspecs:
                if (not a.is_pointer) and is_length_name(a.name) and a.base_type in (
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


def _parse_structs_with_pycparser(cdef: str) -> dict[str, StructSpec]:
    parser = c_parser.CParser()
    ast = parser.parse(cdef)
    structs: dict[str, StructSpec] = {}
    for ext in ast.ext:
        if isinstance(ext, c_ast.Typedef) and isinstance(ext.type, c_ast.TypeDecl):
            if isinstance(ext.type.type, c_ast.Struct):
                struct = ext.type.type
                if struct.decls is None:
                    continue
                fields: list[StructField] = []
                for decl in struct.decls:
                    ctype, is_ptr, is_const, arr_len = _ctype_from_decl(decl.type)
                    base = base_type_from_ctype(ctype)
                    try:
                        numpy_dtype_for_base_type(base)
                    except TypeError:
                        continue
                    fields.append(
                        StructField(
                            name=decl.name,
                            raw_ctype=ctype,
                            base_type=base,
                            is_pointer=is_ptr,
                            is_const=is_const,
                            array_len=arr_len,
                        )
                    )
                if fields:
                    structs[ext.name] = StructSpec(name=ext.name, fields=fields)
    return structs


# ---------- regex fallbacks --------------------------------------------------


def _parse_functions_regex(cdef: str) -> dict[str, FuncSpec]:
    text = cdef
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = strip_restrict_keywords(text)
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

        argspecs: list[ArgSpec] = []
        if arglist and arglist != "void":
            for raw_arg in re.split(r"\s*,\s*", arglist):
                raw_arg = raw_arg.strip()
                if not raw_arg:
                    continue
                m_arg = re.match(r"(.+?)\s*([A-Za-z_]\w*)$", raw_arg)
                if not m_arg:
                    continue
                ctype = m_arg.group(1).strip()
                name = m_arg.group(2)
                is_pointer = "*" in ctype
                is_const = "const" in ctype
                base = base_type_from_ctype(ctype)
                argspecs.append(
                    ArgSpec(
                        name=name,
                        raw_ctype=ctype,
                        base_type=base,
                        is_pointer=is_pointer,
                        is_const=is_const,
                    )
                )
        for a in argspecs:
            if (not a.is_pointer) and is_length_name(a.name) and a.base_type in (
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


def _parse_structs_regex(cdef: str) -> dict[str, StructSpec]:
    structs: dict[str, StructSpec] = {}
    text = strip_restrict_keywords(cdef)
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
            line = re.sub(r"/\*.*?\*/", "", line).strip()
            m_field = re.match(r"(.+?)\s+([A-Za-z_]\w*)(\s*\[(\d+)\])?$", line)
            if not m_field:
                continue
            raw_ctype = m_field.group(1).strip()
            fname = m_field.group(2)
            array_len = int(m_field.group(4)) if m_field.group(4) else None
            is_pointer = "*" in raw_ctype
            is_const = "const" in raw_ctype
            base = base_type_from_ctype(raw_ctype)
            try:
                numpy_dtype_for_base_type(base)
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
