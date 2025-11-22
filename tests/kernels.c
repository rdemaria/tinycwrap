#include <stddef.h>

double dot(const double *restrict x, const double *restrict y, int len_x)
/* Return dot product between x and y */
{
    double acc = 0.0;
    for (int i = 0; i < len_x; ++i)
        acc += x[i] * y[i];
    return acc;
}

void scale(double *restrict x, double alpha, int len_x, double *restrict out_x)
/* Scale vector x by alpha and store the result in out */
{
    for (int i = 0; i < len_x; ++i)
        out_x[i] = alpha * x[i];
}

void cross(const double *restrict a, const double *restrict b, double *restrict out)
/* Compute the cross product between 3D vectors a and b, store the result in out */
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
