#include <math.h>

typedef struct {
    double x;
    double y;
} Point2D;

double point_norm(const Point2D *point)
/* Return sqrt(x^2 + y^2) for a Python-created Point2D. */
{
    return sqrt(point->x * point->x + point->y * point->y);
}

void translate_point(Point2D *point, double dx, double dy)
/* Mutate a Point2D in place. */
{
    point->x += dx;
    point->y += dy;
}

void midpoint(const Point2D *a, const Point2D *b, Point2D *out_point)
/* Fill out_point with the midpoint between a and b. */
{
    out_point->x = 0.5 * (a->x + b->x);
    out_point->y = 0.5 * (a->y + b->y);
}

void make_rectangle(double x0, double y0, double width, double height, Point2D *out_points)
/* Fill out_points with rectangle corners in clockwise order.

Contract: len(out_points)=4;
*/
{
    out_points[0].x = x0;
    out_points[0].y = y0;

    out_points[1].x = x0 + width;
    out_points[1].y = y0;

    out_points[2].x = x0 + width;
    out_points[2].y = y0 + height;

    out_points[3].x = x0;
    out_points[3].y = y0 + height;
}
