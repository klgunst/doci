#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "lapack.h"
#include "debug.h"
#include "davidson.h"

/* For algorithm see http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf, algorithm 12.1 */
int davidson(void* data, double* result, double* energy, int max_vectors, int keep_deflate, \
    double davidson_tol, void (*matvec)(double*, double*, void*), double* diagonal, int basis_size,\
    int max_its){

  int its, m, ONE, dims, INFO, LWORK, cnt_matvecs;
  double *V, *VA, *vec_t, *sub_matrix, *eigv, *eigvalues, *WORK, residue_norm, d_energy;
  char JOBZ, UPLO;
  struct timeval t_start, t_end;
  long long t_elapsed;
  double d_elapsed;

  /* LAPACK initialization */
  JOBZ = 'V';
  UPLO = 'U';
  ONE = 1;
  
  its = 0;
  m = 0;
  residue_norm = davidson_tol * 10;
  d_energy = davidson_tol * 10;
  cnt_matvecs = 0;

  /* store vectors and matrix * vectors */
  V = (double *) malloc( sizeof(double)* basis_size * max_vectors );
  VA = (double *) malloc( sizeof(double)* basis_size * max_vectors );
  vec_t = (double *) malloc(basis_size * sizeof(double)); /* vec_t and residue vector */
  dcopy_(&basis_size, result, &ONE, vec_t, &ONE);

  /* Projected problem */
  sub_matrix = (double *) malloc(sizeof(double)* max_vectors * max_vectors);
  eigv = (double *) malloc(sizeof(double)* max_vectors * max_vectors);
  eigvalues = (double *) malloc(sizeof(double)* max_vectors);
  LWORK = max_vectors * 3 - 1;
  WORK = (double *) malloc(LWORK * sizeof(double));

  gettimeofday(&t_start, NULL);

  printf("IT\tINFO\tRESIDUE\tENERGY\n");
  while( (residue_norm > davidson_tol) && (its++ < max_its) ){
    new_search_vector(V, vec_t, basis_size, m);
    matvec(V + m * basis_size, VA + m * basis_size, data); /* only here expensive matvec needed */
    cnt_matvecs ++;
    expand_submatrix(sub_matrix, V, VA, max_vectors, basis_size, m);
    m++;
    dims = m * max_vectors;
    dcopy_(&dims, sub_matrix, &ONE, eigv, &ONE);
    dsyev_(&JOBZ, &UPLO, &m, eigv, &max_vectors, eigvalues, WORK, &LWORK, &INFO);

    if( m == max_vectors ){   /* deflation */
      deflate(sub_matrix, V, VA, max_vectors, basis_size, keep_deflate, eigvalues, eigv, matvec, \
          data);
      cnt_matvecs += keep_deflate;
      m = keep_deflate;
      dims = m * max_vectors;
      dcopy_(&dims, sub_matrix, &ONE, eigv, &ONE);
      dsyev_(&JOBZ, &UPLO, &m, eigv, &max_vectors, eigvalues, WORK, &LWORK, &INFO);
    }
    residue_norm = calculate_residue(vec_t, result, eigv, eigvalues[0], V, VA, basis_size, m);
    d_energy = *energy - eigvalues[0];
    *energy = eigvalues[0];
    printf("%d\t%d\t%e\t%f\n", its, INFO, residue_norm, eigvalues[0]);
    create_new_vec_t(vec_t, diagonal, eigvalues[0], basis_size);
  }

  if(its <=max_its) printf("Davidson converged in %d iterations with %e residue and d_energy %e.\n"\
      , its, residue_norm, d_energy);
  else printf("Davidson didn't converge within %d iterations! d_energy is %e.\n", max_its,d_energy);

  gettimeofday(&t_end, NULL);
  t_elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000000LL + t_end.tv_usec - t_start.tv_usec;
  d_elapsed = t_elapsed*1e-6;
  printf("elapsed time : %lf sec\n", d_elapsed);
  d_elapsed /= cnt_matvecs;
  printf("average time per matvec : %lf sec\n", d_elapsed);

  *energy = eigvalues[0];
  free(V);
  free(VA);
  free(vec_t);

  free(sub_matrix);
  free(eigv);
  free(eigvalues);
  free(WORK);

  return (its <= max_its);
}

void new_search_vector(double* V, double* vec_t, int basis_size, int m){
  int ONE, i, reortho;
  double *Vi, a;

  ONE = 1;
  Vi = V;
  for( i = 0 ; i < m ; i++ ){
    a = - ddot_(&basis_size, Vi, &ONE, vec_t, &ONE);
    daxpy_(&basis_size, &a, Vi, &ONE, vec_t, &ONE);
    Vi += basis_size;
  }
  a = dnrm2_(&basis_size, vec_t, &ONE);
  a = 1/a;
  dscal_(&basis_size, &a, vec_t, &ONE);

  /**
   * Reorthonormalize if new V is wrong ( eg due to vec_t was very close to a certain V and
   * and numerical errors dont assure orthogonalization )
   */
  reortho = 1;
  while(reortho){
    Vi = V;
    reortho = 0;
    for(i = 0 ; i < m ; i++){
      a = - ddot_(&basis_size, Vi, &ONE, vec_t, &ONE);
      daxpy_(&basis_size, &a, Vi, &ONE, vec_t, &ONE);
      Vi += basis_size;
      if(abs(a) > 1e-12){
        reortho = 1;
        printf("value of a[%d] = %e\n", i, a);
      }
    }
    a = dnrm2_(&basis_size, vec_t, &ONE);
    a = 1/a;
    dscal_(&basis_size, &a, vec_t, &ONE);
  }

  dcopy_(&basis_size, vec_t, &ONE, Vi, &ONE);
}

void expand_submatrix(double* submatrix, double* V, double* VA, int max_vectors, int basis_size, \
                      int m){
  double *Vm, *VAm ,D_ZERO, D_ONE;
  int I_ONE;
  char TRANS;

  I_ONE = 1;
  D_ONE = 1.0;
  D_ZERO = 0.0;
  TRANS = 'T';

  Vm = V + basis_size * m;
  VAm = VA + basis_size * m;

  dgemv_(&TRANS, &basis_size, &m, &D_ONE, V, &basis_size, VAm, &I_ONE, &D_ZERO,\
      submatrix + m*max_vectors, &I_ONE);
  submatrix[m*max_vectors + m] = ddot_(&basis_size, Vm, &I_ONE, VAm, &I_ONE);

  /* lower triangular part is not needed 
  dgemv_(&TRANS, &basis_size, &m, &D_ONE, VA, &basis_size, Vm, &I_ONE, &D_ZERO, submatrix + m, \
      &max_vectors);
  */

  /*
  for(cnt = m ; cnt < m+1 ; cnt ++){
    submatrix[m*max_vectors + cnt] = ddot_(&basis_size, V + cnt*max_vectors, &I_ONE, VAm, &I_ONE);
  }
  
  for(cnt = 0 ; cnt < m+1 ; cnt ++){
    submatrix[m*max_vectors + cnt] = 0;
    for(cnt2 = 0 ; cnt2 < basis_size ; cnt2 ++) {
    printf("%f\n", submatrix[m*max_vectors + cnt]);
    submatrix[m*max_vectors + cnt] += V[cnt*max_vectors + cnt2] * VAm[cnt2];
    }
  }
  */
}

double calculate_residue(double* residue, double *result, double* eigv, double theta, \
                        double* V, double* VA, int basis_size, int m){
  double mtheta, D_ONE, D_ZERO, residue_norm;
  int I_ONE;
  char NTRANS;
  D_ZERO = 0;
  D_ONE = 1;
  I_ONE = 1;
  NTRANS = 'N';

  mtheta = -theta;
  dgemv_(&NTRANS, &basis_size, &m, &D_ONE, V, &basis_size, eigv, &I_ONE, &D_ZERO, result, &I_ONE);
  dgemv_(&NTRANS,&basis_size,&m, &D_ONE, VA, &basis_size, eigv, &I_ONE, &D_ZERO, residue, &I_ONE);
  daxpy_(&basis_size, &mtheta, result, &I_ONE, residue, &I_ONE);
  residue_norm = dnrm2_(&basis_size, residue, &I_ONE);
  return residue_norm;
}

void create_new_vec_t(double* residue, double* diagonal, double theta, int size){
  /* quick implementation */
  int i;
  double cutoff = 1e-6;
  for( i = 0 ; i < size ; i++)
    if(abs(diagonal[i] - theta) > cutoff)
      residue[i] = residue[i] / abs(diagonal[i] - theta);
    else{
      residue[i] = residue[i] / cutoff;
    }
}

void deflate(double* sub_matrix, double* V, double* VA, int max_vectors, int basis_size, \
              int keep_deflate, double* eigvalues, double* eigv, void (*matvec)(double*, \
                double*, void*), void* data){
  double *new_result, D_ZERO, D_ONE;
  int I_ONE, size_x_deflate, i;
  char NTRANS;
  D_ZERO = 0;
  D_ONE = 1;
  I_ONE = 1;
  NTRANS = 'N';

  size_x_deflate = basis_size * keep_deflate;

  new_result = (double *) malloc(basis_size * keep_deflate * sizeof(double));

  dgemm_(&NTRANS, &NTRANS, &basis_size, &keep_deflate, &max_vectors, &D_ONE, V, &basis_size, eigv, \
          &max_vectors, &D_ZERO, new_result , &basis_size);
  dcopy_(&size_x_deflate, new_result, &I_ONE, V, &I_ONE);

  for(i = 0 ; i < keep_deflate; i ++)
    matvec(V + i * basis_size, VA + i * basis_size, data); /* only here expensive matvec needed */

  /**
   * I think doing this makes it unstable
   *
   * dgemm_(&NTRANS, &NTRANS, &basis_size, &keep_deflate, &max_vectors, &D_ONE, VA, &basis_size, eigv,\
   *       &max_vectors, &D_ZERO, new_result , &basis_size);
   *
   * dcopy_(&size_x_deflate, new_result, &I_ONE, VA, &I_ONE);
   *
   * No, it is not this, but maybe keep this though, you never know it also fucks up.
   *
   * Doing matvecs again is more expensive, but less prone to numerical errors accumulating.
   */

  free(new_result);

  for( i = 0 ; i < keep_deflate; i++)
    expand_submatrix(sub_matrix, V, VA, max_vectors, basis_size, i);
}
