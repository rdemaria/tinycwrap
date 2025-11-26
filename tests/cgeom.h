#ifndef CGEOM_C
#define CGEOM_C

/* 2D primitives */

typedef struct G2DPoint {
    double x;
    double y;
} G2DPoint;


typedef struct {
    int type; /* 0=line, 1=arc */
    double data[8]; 
} G2DSegment;


typedef struct {
    G2DPoint *points;
    int len_points;
} G2DPoints;

double geom2d_polygon_length(const G2DPoint *points, int len_points);


#endif /* CGEOM_C */
