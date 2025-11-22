import numpy as np
from cmodule import CModule

cm = CModule("kernels.c")   # cdef auto-generated, wrappers auto-created

x = np.arange(10, dtype=np.float64)
y = np.ones_like(x)

print(cm.dot(x, y))   # -> 45.0
help(cm.dot)          # shows your C comment as docstring


