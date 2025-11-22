import numpy as np
from tinycwrap import CModule

cm = CModule("kernels.c")   # cdef auto-generated, wrappers auto-created

x = np.arange(10, dtype=np.float64)
y = np.ones_like(x)

print(cm.dot(x, y))   # -> 45.0
print(cm.dot.__doc__)          # shows your C comment as docstring
print(cm.dot.__source__)          # shows your C comment as docstring

print(cm.scale(x, 2.2)) 
out = np.empty_like(x)
print(cm.scale(x, 3.3, out_x=out))
