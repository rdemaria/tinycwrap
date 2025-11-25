#include <math.h>
#include <stdlib.h>

#include "base.h"


double geom2d_norm(double x, double y)
/* Get the norm of a 2D vector */
{
    return sqrt(x * x + y * y);
}

double geom2d_points_distance(double x1, double y1, double x2, double y2)
/* Get the distance between two 2D points */
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    return geom2d_norm(dx, dy);
}