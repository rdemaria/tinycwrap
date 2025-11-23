#include <stddef.h>
#include <math.h>
#include "test_kernels.h"

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

void cross(const double *restrict a, const double *restrict b, double *restrict out)
/* Compute the cross product between 3D vectors a and b, store the result in out */
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

double complex_magnitude(const ComplexPair *z)
/* Return sqrt(re^2 + im^2) */
{
    return sqrt(z->real * z->real + z->imag * z->imag);
}

double kinetic_energy(const Particle *p, int len_p)
/* Sum 0.5*|v|^2 over particles */
{
    double sum = 0.0;
    for (int i = 0; i < len_p; ++i) {
        double vx = p[i].vel[0];
        double vy = p[i].vel[1];
        double vz = p[i].vel[2];
        sum += 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    return sum;
}


void split_vectors(const double *restrict inp, int len_inp, double *restrict out1, double *restrict out2)
/* Split even/odd elements of inp into out1/out2 */
{
    for (int i = 0; i < len_inp; ++i) {
        if (i % 2 == 0)
            out1[i / 2] = inp[i];
        else
            out2[i / 2] = inp[i];
    }
}

void make_particles(double speed, Particle *out_p, int len_p)
/* Fill particles with unit positions and uniform speed on x */
{
    for (int i = 0; i < len_p; ++i) {
        out_p[i].pos[0] = 1.0;
        out_p[i].pos[1] = 1.0;
        out_p[i].pos[2] = 1.0;
        out_p[i].vel[0] = speed;
        out_p[i].vel[1] = 0.0;
        out_p[i].vel[2] = 0.0;
    }
}
