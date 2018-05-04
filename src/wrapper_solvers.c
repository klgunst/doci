#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wrapper_solvers.h"
#include "debug.h"
#include "CG.h"
#include "davidson.h"

int sparse_eigensolve(double* vec, int dims, double* energy, void (*matvec)(double*, double*, 
      void* ), double* diagonal, double *M, double tol, int max_its, char solver[], 
    int davidson_keep, int davidson_max_vec, void* dat )
{
  int res = -1;
  if(strcmp(solver, "D") == 0)
    res = davidson(vec, energy, davidson_max_vec, davidson_keep, tol, matvec, diagonal, dims,\
                    max_its, dat );
  else if(strcmp(solver, "CG") == 0)
    res = ray_q(matvec, M, vec, dims, energy, tol, max_its, NULL, dat );
  else if(strcmp(solver, "CGP") == 0)
    res = ray_q(matvec, M, vec, dims, energy, tol, max_its, diagonal, dat );

  return res;
}
