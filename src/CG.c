#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "lapack.h"
#include "debug.h"
#include "CG.h"

/* For algorithm see http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter13.pdf, algorithm 13.1 */
int ray_q(void (*matvec)(double*, double*, void*), void *data, double* M, double* vec,\
    int size, double *energy, double cg_tol, int max_its, double* precond){
  /* here i still assume M == NULL, no preconditioner and generalized eig
   * implemented yet */
  assert(M == NULL);

  int its, I_ONE, I_TWO, ITYPE, LWORK, INFO, cnt;
  double rho, minrho, normq_g, quot_normq, normq_g_prev, cutoff, d_energy, normq;
  double D_TWO, D_ONE, D_ZERO, mONE;
  double *x, *p, *Ax, *Ap, *g, *A, *B, *eigv, *WORK, *precond_inv;
  char JOBZ, TRANS, NTRANS, UPLO;
  struct timeval t_start, t_end;
  long long t_elapsed;
  double d_elapsed;

  /* LAPACK init */
  D_TWO = 2;
  D_ONE = 1;
  D_ZERO = 0;
  mONE = -1;
  I_TWO = 2;
  I_ONE = 1;
  ITYPE = 1;
  LWORK = 3 * I_TWO -1;
  JOBZ = 'V';
  TRANS = 'T';
  NTRANS = 'N';
  UPLO = 'U';

  x = (double*) malloc(size * 2 * sizeof(double));
  Ax = (double*) malloc(size * 2 * sizeof(double));
  g = (double*) malloc(size * sizeof(double));
  A = (double*) malloc(2 * 2 * sizeof(double));
  B = (double*) malloc(2 * 2 * sizeof(double));
  eigv = (double*) malloc(2 * sizeof(double));
  WORK = (double*) malloc(LWORK * sizeof(double));
  
  precond_inv = (double*) malloc(size * sizeof(double));
  cutoff = 1e-12;
  if(precond != NULL){
    for(cnt = 0 ; cnt < size; cnt++)
      if(abs(precond[cnt]) > cutoff)
        precond_inv[cnt] = 1/precond[cnt];
      else
        precond_inv[cnt] = 1/cutoff;
  }

  p = x + size;
  Ap = Ax + size;


  its = 0;
  gettimeofday(&t_start, NULL);
  DPRINTL("\n");
  DPRINT("IT\tINFO\tRESIDUE\tENERGY\n");

  dcopy_(&size, vec, &I_ONE, x, &I_ONE);
  matvec(x, Ax, data);
  rho = ddot_(&size, Ax, &I_ONE, x, &I_ONE);
  minrho = -rho;
  dcopy_(&size, Ax, &I_ONE, g, &I_ONE);
  daxpy_(&size, &minrho, x, &I_ONE, g, &I_ONE);
  dscal_(&size, &D_TWO, g, &I_ONE);
  normq = ddot_(&size, g, &I_ONE, g, &I_ONE);

  if(precond != NULL)
    for(cnt = 0 ; cnt < size; cnt++) g[cnt] = g[cnt] * precond_inv[cnt];
  normq_g = ddot_(&size, g, &I_ONE, g, &I_ONE);

  printf("%d\t%d\t%e\t%f\n", its, 0, normq_g, rho);

  while( normq > cg_tol && its < max_its){
    if(its != 0){
      quot_normq = normq_g / normq_g_prev;
      dscal_(&size, &quot_normq, p, &I_ONE);
      daxpy_(&size, &mONE, g, &I_ONE, p, &I_ONE);
    }
    else{
      for(cnt = 0 ; cnt < size; cnt++) p[cnt] = -g[cnt];
    }
    
    /* eigenvalue problem */
    matvec(p, Ap, data);
    dgemm_(&TRANS, &NTRANS, &I_TWO, &I_TWO, &size, &D_ONE, x, &size, Ax, &size, &D_ZERO, A, &I_TWO);
    dgemm_(&TRANS, &NTRANS, &I_TWO, &I_TWO, &size, &D_ONE, x, &size,  x, &size, &D_ZERO, B, &I_TWO);
    dsygv_(&ITYPE, &JOBZ, &UPLO, &I_TWO, A, &I_TWO, B, &I_TWO, eigv, WORK, &LWORK, &INFO);
    
    dscal_(&size, A, x, &I_ONE);
    daxpy_(&size, A+1, p, &I_ONE, x, &I_ONE);
    
    /*
    matvec(x, Ax, data);
    */
    dscal_(&size, A, Ax, &I_ONE);
    daxpy_(&size, A+1, Ap, &I_ONE, Ax, &I_ONE);

    d_energy = rho - eigv[0];
    rho = eigv[0];
    minrho = -rho;
    dcopy_(&size, Ax, &I_ONE, g, &I_ONE);
    daxpy_(&size, &minrho, x, &I_ONE, g, &I_ONE);
    dscal_(&size, &D_TWO, g, &I_ONE);
    normq = ddot_(&size, g, &I_ONE, g, &I_ONE);
    if(precond != NULL)
      for(cnt = 0 ; cnt < size; cnt++) g[cnt] = g[cnt] * precond_inv[cnt];

    normq_g_prev = normq_g;
    normq_g = ddot_(&size, g, &I_ONE, g, &I_ONE);
    printf("%d\t%d\t%e\t%f\n", its, INFO, normq_g, rho);
    its++;
  }
  if(its <=  max_its) printf("CG converged in %d iterations with %e residue and d_energy %e.\n"\
      , its, normq_g, d_energy);
  else printf("CG didn't converge within %d iterations! d_energy is %e.\n", max_its,d_energy);

  gettimeofday(&t_end, NULL);
  t_elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000000LL + t_end.tv_usec - t_start.tv_usec;
  d_elapsed = t_elapsed*1e-6;
  printf("elapsed time : %lf sec\n", d_elapsed);
  d_elapsed /= its + 1;
  printf("average time per matvec : %lf sec\n", d_elapsed);



  dcopy_(&size, x, &I_ONE, vec, &I_ONE);
  *energy = rho;

  free(x);
  free(Ax);
  free(g);
  free(A);
  free(B);
  free(eigv);
  free(WORK);
  free(precond_inv);

  return (its <= max_its);
}
