#ifndef CG_H 
# define CG_H

int ray_q(void (*matvec)(double*, double*, void* ), double*, double*, int, double*, \
    double, int, double*, void* );

#endif
