#include <stddef.h>

double dot(const double *x, const double *y, int len_x)
/* Return dot product between x and y */
{
    double acc = 0.0;
    for (int i = 0; i < len_x; ++i)
        acc += x[i] * y[i];
    return acc;
}


