from pathlib import Path

from tinycwrap import CModule

here = Path(__file__).resolve().parent
cm = CModule(here / "points.c")

# Python constructs C-compatible structs from the generated Point2D class.
a = cm.Point2D(x=3.0, y=4.0)
b = cm.Point2D(x=7.0, y=10.0)

print(a)
print(cm.point_norm(a))

# Non-const struct pointers are passed by reference and can be mutated by C.
cm.translate_point(a, dx=1.0, dy=-2.0)
print(a)

# Leaving an out_* struct argument at None lets TinyCWrap allocate storage,
# pass it to C, and return the filled Point2D object to Python.
mid = cm.midpoint(a, b)
print(mid)

# Struct arrays are returned as Point2DArray wrappers around NumPy storage.
rectangle = cm.make_rectangle(x0=1.0, y0=2.0, width=5.0, height=3.0)
print(rectangle)
print(rectangle.x)
print(rectangle.y)
print(rectangle[1])

from_array_like = cm.Point2DArray([(0.0, 0.0), (1.0, 1.0)])
from_fields = cm.Point2DArray(x=[2.0, 3.0], y=[4.0, 5.0])
print(from_array_like)
print(from_fields.x)
print(from_fields.y)
