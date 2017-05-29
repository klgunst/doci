#ifndef CG_H 
# define CG_H

int ray_q(void (*matvec)(double*, double*, void*), void*, double*, double*, int, double*, \
    double, int, double*);

#endif
