#pragma once

typedef struct {
    double real;
    double imag;
} ComplexPair;

typedef struct {
    double pos[3];
    double vel[3];
} Particle;

double dot(const double *restrict x, const double *restrict y, int len_x);
void scale(double *restrict x, double alpha, int len_x, double *restrict out_x);
void mat_add(const double *restrict a, const double *restrict b, int n, int m, double *restrict out);
void cross(const double *restrict a, const double *restrict b, double *restrict out);
double complex_magnitude(const ComplexPair *z);
double kinetic_energy(const Particle *p, int len_p);
void split_vectors(const double *restrict inp, int len_inp, double *restrict out1, double *restrict out2);
void make_particles(double speed, Particle *out_p, int len_p);
void merge_sorted(const double * a, const double * b, int len_a, int len_b, double *out, int *out_len);
double *alloc_random_array(int *out_len);
