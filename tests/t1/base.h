#ifndef CGEOM_BASE_H
#define CGEOM_BASE_H
/* Basic geometric types and functions */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double x;
    double y;
} G2DPoint;

typedef struct {
    double x;
    double y;
    double z;
} G3DPoint;

double geom2d_norm(double x, double y);
double geom2d_points_distance(double x1, double y1, double x2, double y2);

#endif /* CGEOM_BASE_H */