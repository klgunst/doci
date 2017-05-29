#ifndef DAVIDSON_H
# define DAVIDSON_H

int davidson(void*, double*, double*, int, int, double, void (*matvec)(double*, double*, void*), \
              double*, int, int);

void new_search_vector(double*, double*, int, int);

void expand_submatrix(double*, double*, double*, int, int, int);

double calculate_residue(double*, double*, double*, double, double*, double*, int, int);

void create_new_vec_t(double*, double*, double, int);

void deflate(double*, double*, double*, int, int, int, double*, double*);

#endif
