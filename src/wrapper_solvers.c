#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "CG.h"
#include "davidson.h"

int sparse_eigensolve(double* vec,int dims, double* energy,void (*matvec)(double*, double*, void*),\
    void* data, double* diagonal, double *M, double tol,int max_its, char solver[],\
    int davidson_keep, int davidson_max_vec){
  int res;
  res = -1;
  if(strcmp(solver, "D") == 0)
    res = davidson(data, vec, energy, davidson_max_vec, davidson_keep, tol, matvec, diagonal, dims,\
                    max_its);
  else if(strcmp(solver, "CG") == 0)
    res = ray_q(matvec, data, M, vec, dims, energy, tol, max_its, NULL);
  else if(strcmp(solver, "CGP") == 0)
    res = ray_q(matvec, data, M, vec, dims, energy, tol, max_its, diagonal);

  return res;
}
