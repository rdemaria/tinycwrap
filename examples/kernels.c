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
