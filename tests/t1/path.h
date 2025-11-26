#ifndef CGEOM_PATH_H
#define CGEOM_PATH_H

#include "base.h"

/* 2D PATH made of segments  
type: 0=line, 1=arc, 2=ellipse arc, 3=quadratic bezier, 4=cubic bezier
data:
 line: x1,y1,x2,y2
 arc: cx,cy,start_angle,end_angle
 ellipse arc: cx,cy,rx,ry,rotation,start_angle,end_angle
 quadratic bezier: x1,y1,x2,y2,cx,cy
 cubic bezier: x1,y1,x2,y2,cx1,cy1,cx2,cy2
*/


typedef struct {
    int type; /* 0=line, 1=arc */
    double data[8]; 
} G2DSegment;

typedef struct {
    G2DSegment *segments;
    int len_segments;
} G2DPath;

void geom2d_line_segment_from_start_length(double x0, double y0, double dx, double dy, double length, G2DSegment *out);
double geom2d_line_segment_get_length(const G2DSegment *seg);
double geom2d_line_segment_get_points_at_steps(const G2DSegment *seg, const double *steps, int len_points, G2DPoint *out_points);

void geom2d_arc_segment_from_start_length_angle(double x0, double y0, double dx, double dy, double length, double angle, G2DSegment *out);
void geom2d_arc_segment_get_pd_at_length(const G2DSegment *seg, double at, double *out_x, double *out_y, double *out_dx, double *out_dy);
double geom2d_arc_segment_get_length(const G2DSegment *seg);
double geom2d_arc_segment_get_points_at_steps(const G2DSegment *seg, const double *steps, int len_points, G2DPoint *out_points);

double geom2d_ellipse_segment_get_length(const G2DSegment *seg);

void geom2d_rectangle_to_path(double halfwidth, double halfheight, G2DSegment *out_segments);
void geom2d_ellipse_to_path(double rx, double ry, G2DSegment *out_segment);
void geom2d_rectellipse_to_path(double halfwidth, double halfheight, double rx, double ry, G2DSegment *out_segments, int *out_len);

int geom2d_path_get_len_steps(const G2DSegment *segments, int len_segments, double ds_min);
double geom2d_path_get_length(const G2DSegment *segments, int len_segments);
void geom2d_path_get_steps(const G2DSegment *segments, int len_segments, double ds_min, double *out_steps);


#endif /* CGEOM_PATH_H */
