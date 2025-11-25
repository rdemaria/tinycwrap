#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "cgeom.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void geom2d_circle_get_n_points(double cx, double cy, double r, int len_points, G2DPoint *out_points)
/* Get n points on the circumference of a circle */
{
    if (len_points <= 0) {
        return;
    }

    double angle_step = 2.0 * M_PI / len_points;

    for (int i = 0; i < len_points; i++) {
        double angle = i * angle_step;
        out_points[i].x = cx + r * cos(angle);
        out_points[i].y = cy + r * sin(angle);
    }
}

void geom2d_circle_get_points(double cx, double cy, double r, G2DPoint *out_points)
/* Get 101 points on the circumference of a circle

Contract: len(out_points)=101;
*/
{
    int len_points = 101;
    double angle_step = 2.0 * M_PI / len_points;
    for (int i = 0; i < len_points; i++) {
        double angle = i * angle_step;
        out_points[i].x = cx + r * cos(angle);
        out_points[i].y = cy + r * sin(angle);
    }
}

double geom2d_norm(double x, double y)
/* Get the norm of a 2D vector */
{
    return sqrt(x*x + y*y);
}

void geom2d_line_segment_from_start_length(double x0, double y0, double dx, double dy, double length, G2DSegment *out)
/* Get line data from starting point, direction (assuming dx,dy have norm=1) and length

*/
{
    double ux = dx;
    double uy = dy;
    out->data[0] = x0;
    out->data[1] = y0;
    out->data[2] = x0 + ux * length;
    out->data[3] = y0 + uy * length;
    out->type = 0; /* line */
}

