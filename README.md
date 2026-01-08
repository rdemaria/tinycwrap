# tinycwrap

TinyCWrap is a lightweight helper around CFFI that:

- compiles one or more C sources into a shared library,
- auto-generates the `cdef` from your function/struct declarations,
- builds NumPy-friendly Python wrappers that take/return arrays and structs,
- optionally hot-reloads when the C file changes (inside IPython).

## Quick start

```bash
pip install tinycwrap
```

Write a small C file (only non-`static` functions are exported):

```c
/* kernels.c */
double dot(const double *x, const double *y, int len_x)
/* Return dot product between x and y
   Contract: len_x=len(x);
*/
{
    double acc = 0.0;
    for (int i = 0; i < len_x; ++i)
        acc += x[i] * y[i];
    return acc;
}
```

Wrap and call it from Python:

```python
import numpy as np
from tinycwrap import CModule

cm = CModule("kernels.c")  # builds the shared library, creates wrappers

x = np.arange(5, dtype=np.float64)
y = np.ones_like(x)

cm.dot(x, y)           # -> 10.0, len_x auto-filled by the contract
print(cm.dot.__doc__)  # docstring comes from the C comment
```

See `examples/` and `tests/` for more involved cases.

## C coding conventions

TinyCWrap infers how to build wrappers from simple C conventions:

- **Inputs vs outputs**
  - `const T *arg` -> input NumPy array.
  - `T *arg` -> output/in-place array (prefer naming it `out_*` when possible).
  - `T arg[N]` in the signature -> fixed-size array (input if `const`, otherwise output).
  - Plain scalars (`double`, `int`, ...) -> Python scalars.
- **Length parameters**: integers such as `len_x`, `n`, `size_x` can be auto-filled if you declare a contract (see below). Otherwise pass them explicitly from Python.
- **Docstrings**: the block comment immediately after the function header becomes the Python docstring.
- **Structs**: `typedef struct { ... } Name;` definitions in your headers/sources become Python classes with a `.dtype` and NumPy-backed storage.
- **Compilation**: extra sources can be passed (`CModule("main.c", "helper.c")`), and extra include dirs via `include_dirs=[...]`. Default compiler flags are `-O3 -shared -fPIC -march=native -mtune=native` plus the NumPy include path.

## Contracts: tell the wrapper how to size things

Contracts are declared inside the doc comment with `Contract:` (or `Contracts:`). Separate multiple rules with semicolons. Supported forms:

- `len_x=len(x)` — mark `len_x` as the length of array `x`. If you omit `len_x` in Python, the wrapper fills it.
- `shape(out)=n,m` — allocate `out` with shape `(n, m)`. You can define `n,m=shape(a)` to capture the shape of an input first.
- `len(out)=len_x` — allocate a 1D output array.
- `postlen(out)=out_len` — slice the output after the call using an integer pointer result `out_len` (useful when the C code writes less than the allocated length).
- `own(return)` — the returned pointer is owned by the caller; the wrapper will `free` it after copying to NumPy (requires a `len(return)=...` contract).

Examples pulled from the test suite:

```c
void mat_add(const double *a, const double *b, int n, int m, double *out)
/* Elementwise addition
   Contract: n,m=shape(a); shape(out)=n,m;
*/
```

```c
void merge_sorted(const double *a, const double *b, int len_a, int len_b,
                  double *out, int *out_len)
/* Merge unique sorted values
   Contract: len_a=len(a); len_b=len(b);
             len(out)=len_a+len_b; postlen(out)=out_len;
*/
```

```c
double *alloc_random_array(int *out_len)
/* Return a freshly malloc'ed array
   Contract: len(return)=out_len; own(return);
*/
```

Rules are parsed case-insensitively; whitespace does not matter.

## Python wrapper behavior

- **Automatic allocation**: any `out_*` argument defaults to `None` in Python; TinyCWrap allocates it based on contracts, fixed array sizes, or by matching the shape of a related input (`out_x` matches `x` when no contract is present).
- **Struct pointer outputs**: non-const struct pointers are treated as in/out; when you pass an object/array explicitly, it is mutated in place and not returned. When you pass `None`, TinyCWrap allocates and returns the struct (or struct array).
- **Optional length arguments**: integer length parameters inferred from contracts default to `None` in the wrapper signature. If you pass them, they are cast to `int`; if not, the expression from the contract is evaluated.
- **Post contracts**: when a contract uses `postlen(...)` or `post shape(...)`, the wrapper slices/reshapes outputs after the C call using the values written by the C function.
- **Scalar pointer outputs**: pointers to integer-like types (e.g., `int *out_len`) are returned as plain Python integers alongside other outputs.
- **Structs**: for `typedef struct` declarations TinyCWrap generates a Python class:
  - fields are accessible as properties backed by a NumPy structured dtype (`Name.dtype`);
  - pointer fields paired with `len_<field>` automatically allocate NumPy arrays when you instantiate the struct with `len_field` or when you pass an array for that field;
  - `_data` holds the underlying structured array, and `Name.zeros(n)` returns an array of that dtype.
- **Reloading**: inside IPython, pass `reload=True` (default) to auto-recompile before each cell if the C sources changed.

## Worked examples

### Two outputs and automatic lengths

```c
void split_vectors(const double *inp, int len_inp,
                   double *out_even, double *out_odd)
/* Split even/odd elements
   Contract: len_inp=len(inp); len(out_even)=len_inp/2; len(out_odd)=len_inp/2;
*/
```

```python
data = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
out_even, out_odd = cm.split_vectors(data)  # lengths inferred, arrays allocated
```

### Structs and struct arrays

```c
typedef struct {
    double real;
    double imag;
} ComplexPair;

double complex_magnitude(const ComplexPair *z);
```

```python
cp = cm.ComplexPair(real=3.0, imag=4.0)
cm.complex_magnitude(cp)  # -> 5.0

# struct arrays are NumPy dtypes; useful when C expects pointers to arrays of structs
pairs = cm.ComplexPair.zeros(2)
pairs["real"] = [1.0, 2.0]
pairs["imag"] = [0.0, -1.0]
```

### Owned return value

```c
double *alloc_random_array(int *out_len)
/* Contract: len(return)=out_len; own(return); */
```

```python
arr, n = cm.alloc_random_array()  # returns NumPy array, frees the C buffer
```

## Tips

- Keep function declarations (or headers) visible at the top level; TinyCWrap scans the C files and headers you pass.
- If a length cannot be inferred from a contract, you must pass it explicitly from Python.
- Use `cm.<func>.__source__` to inspect the wrapper code if something behaves unexpectedly.
- For debugging parsed signatures/contracts call `cm._debug_specs()`.

That is all you need to start turning small C helpers into ergonomic Python callables.
