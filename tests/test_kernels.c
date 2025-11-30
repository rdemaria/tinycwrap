#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include "test_kernels.h"

double dot(const double *restrict x, const double *restrict y, int len_x)
/* Return dot product between x and y

Contract: len_x=len(x)
*/
{
    double acc = 0.0;
    for (int i = 0; i < len_x; ++i)
        acc += x[i] * y[i];
    return acc;
}

void scale(double *restrict x, double alpha, int len_x, double *restrict out_x)
/* Scale vector x by alpha and store the result in out

Contract: len_x=len(x); len(out_x)=len_x;
*/
{
    for (int i = 0; i < len_x; ++i)
        out_x[i] = alpha * x[i];
}

void cross(const double *restrict a, const double *restrict b, double *restrict out)
/* Compute the cross product between 3D vectors a and b, store the result in out 

Contract: len(out)=3;
*/

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
/* Sum 0.5*|v|^2 over particles

Contract: len_p=len(p);
*/
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
/* Split even/odd elements of inp into out1/out2

Contract: len_inp=len(inp); len(out1)=len_inp/2; len(out2)=len_inp/2;
*/
{
    for (int i = 0; i < len_inp; ++i) {
        if (i % 2 == 0)
            out1[i / 2] = inp[i];
        else
            out2[i / 2] = inp[i];
    }
}

void make_particles(double speed, Particle *out_p, int len_p)
/* Fill particles with unit positions and uniform speed on x

Contract: len(out_p)=len_p;
*/
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


double geom2d_norm(double x, double y)
/* Get the norm of a 2D vector */
{
    return sqrt(x*x + y*y);
}

void merge_sorted(const double *restrict a, const double *restrict b, int len_a, int len_b, double *out, int *out_len)
/* Return array with not repeated ascending values from a and b that are assumed to be sorted

Contract: len_a=len(a); len_b=len(b); len(out)=len_a+len_b;
Post-Contract: len(out)=out_len;
*/
{
    int ia = 0, ib = 0, k = 0;
    double last = NAN;
    while (ia < len_a && ib < len_b) {
        double va = a[ia];
        double vb = b[ib];
        double v;
        if (va < vb) {
            v = va;
            ia++;
        } else if (vb < va) {
            v = vb;
            ib++;
        } else {
            v = va;
            ia++;
            ib++;
        }
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    while (ia < len_a) {
        double v = a[ia++];
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    while (ib < len_b) {
        double v = b[ib++];
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    *out_len = k;
}

double *alloc_random_array(int *out_len)
/* Allocate an array with random-ish length

Own: return
Contract: len(return)=out_len
*/
{
    int n = (int)(rand() % 5 + 3); /* length between 3 and 7 */
    double *arr = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        arr[i] = (double)(i + 1);
    *out_len = n;
    return arr;
}
