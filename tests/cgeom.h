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


#endif /* CGEOM_C */
