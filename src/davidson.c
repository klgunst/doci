#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include "lapack.h"
#include "debug.h"
#include "davidson.h"
#include "macros.h"

/* ============================================================================================ */
/* =============================== DECLARATION STATIC FUNCTIONS =============================== */
/* ============================================================================================ */

static void new_search_vector(double* V, double* vec_t, int basis_size, int m);

static void expand_submatrix(double* submatrix, double* V, double* VA, int max_vectors, 
    int basis_size, int m);

static double calculate_residue(double* residue, double *result, double* eigv, double theta, 
    double* V, double* VA, int basis_size, int m);

static void create_new_vec_t(double* residue, double* diagonal, double theta, int size);

static void deflate(double* sub_matrix, double* V, double* VA, int max_vectors, int basis_size, 
    int keep_deflate, double* eigv);

/* ============================================================================================ */

/* For algorithm see http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf, algorithm 12.1 */
int davidson(double* result, double* energy, int max_vectors, int keep_deflate, \
    double davidson_tol, void (*matvec)(double*, double*), double* diagonal, int basis_size,\
    int max_its){

  /* LAPACK initialization */
  char JOBZ = 'V';
  char UPLO = 'U';
  int ONE = 1;
  
  int its = 0;
  int m = 0;
  double residue_norm = davidson_tol * 10;
  double d_energy = davidson_tol * 10;
#ifdef DAVIDIT
  struct timeval t_start, t_end;
  long long t_elapsed;
  double d_elapsed;

  int cnt_matvecs = 0;
  gettimeofday(&t_start, NULL);
  printf("IT\tINFO\tRESIDUE\tENERGY\n");
#endif

  /* Projected problem */
  double *sub_matrix = safe_malloc(max_vectors * max_vectors, double);
  double *eigv = safe_malloc(max_vectors * max_vectors, double);
  double *eigvalues = safe_malloc(max_vectors, double);
  int LWORK = max_vectors * 3 - 1;
  double *WORK = safe_malloc(LWORK, double);

  /* store vectors and matrix * vectors */
  double *V = safe_malloc( basis_size * max_vectors, double );
  double *VA = safe_malloc(  basis_size * max_vectors, double );
  double *vec_t = safe_malloc(basis_size, double); /* vec_t and residue vector */
  dcopy_(&basis_size, result, &ONE, vec_t, &ONE);

  *energy = 0;

  while( (residue_norm > davidson_tol) && (its++ < max_its) ){
    int dims;
    int INFO = 0;
    new_search_vector(V, vec_t, basis_size, m);
    matvec(V + m * basis_size, VA + m * basis_size); /* only here expensive matvec needed */
    expand_submatrix(sub_matrix, V, VA, max_vectors, basis_size, m);
    m++;
    dims = m * max_vectors;
    dcopy_(&dims, sub_matrix, &ONE, eigv, &ONE);
    dsyev_(&JOBZ, &UPLO, &m, eigv, &max_vectors, eigvalues, WORK, &LWORK, &INFO);

    if( m == max_vectors ){   /* deflation */
      deflate(sub_matrix, V, VA, max_vectors, basis_size, keep_deflate, eigv);
      m = keep_deflate;
      dims = m * max_vectors;
      dcopy_(&dims, sub_matrix, &ONE, eigv, &ONE);
      dsyev_(&JOBZ, &UPLO, &m, eigv, &max_vectors, eigvalues, WORK, &LWORK, &INFO);
    }
    residue_norm = calculate_residue(vec_t, result, eigv, eigvalues[0], V, VA, basis_size, m);
      
    d_energy = *energy - eigvalues[0];
    *energy = eigvalues[0];
#ifdef DAVIDIT
    cnt_matvecs++;
    printf("%d\t%d\t%e\t%f\n", its, INFO, residue_norm, eigvalues[0]);
#endif
    create_new_vec_t(vec_t, diagonal, eigvalues[0], basis_size);
  }

#ifdef DAVIDIT
  if(its <= max_its) 
    printf("Davidson converged in %d iterations with %e residue and d_energy %e.\n", its, residue_norm, d_energy);
  else 
    printf("Davidson didn't converge within %d iterations! d_energy is %e.\n", max_its,d_energy);

  gettimeofday(&t_end, NULL);
  t_elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000000LL + t_end.tv_usec - t_start.tv_usec;
  d_elapsed = t_elapsed*1e-6;
  printf("elapsed time : %lf sec\n", d_elapsed);
  d_elapsed /= cnt_matvecs;
  printf("average time per matvec : %lf sec\n", d_elapsed);
#else
  printf("\t\t ** DAVIDSON ITERATIONS : %d\n", its);
#endif

  *energy = eigvalues[0];
  safe_free(V);
  safe_free(VA);
  safe_free(vec_t);

  safe_free(sub_matrix);
  safe_free(eigv);
  safe_free(eigvalues);
  safe_free(WORK);

  return (its >= max_its);
}

/* ============================================================================================ */
/* ================================ DEFINITION STATIC FUNCTIONS =============================== */
/* ============================================================================================ */
static void new_search_vector(double* V, double* vec_t, int basis_size, int m){
  int ONE = 1;
  double *Vi = V;
  double a;
  int i;

  for( i = 0 ; i < m ; i++ ){
    a = - ddot_(&basis_size, Vi, &ONE, vec_t, &ONE);
    daxpy_(&basis_size, &a, Vi, &ONE, vec_t, &ONE);
    Vi += basis_size;
  }
  a = dnrm2_(&basis_size, vec_t, &ONE);
  a = 1/a;
  dscal_(&basis_size, &a, vec_t, &ONE);

#ifdef DEBUG
  {
    /**
     * Reorthonormalize if new V is wrong ( eg due to vec_t was very close to a certain V and
     * and numerical errors dont assure orthogonalization )
     */
    int reortho = 1;
    while(reortho){
      Vi = V;
      reortho = 0;
      for(i = 0 ; i < m ; i++){
        a = - ddot_(&basis_size, Vi, &ONE, vec_t, &ONE);
        daxpy_(&basis_size, &a, Vi, &ONE, vec_t, &ONE);
        Vi += basis_size;
        if(fabs(a) > 1e-10){
          reortho = 1;
          printf("value of a[%d] = %e\n", i, a);
          assert(0);
        }
      }
      a = dnrm2_(&basis_size, vec_t, &ONE);
      a = 1/a;
      dscal_(&basis_size, &a, vec_t, &ONE);
    }
  }
#endif

  dcopy_(&basis_size, vec_t, &ONE, Vi, &ONE);
}

static void expand_submatrix(double* submatrix, double* V, double* VA, int max_vectors, 
    int basis_size, int m){

  int I_ONE = 1;
  double D_ONE = 1.0;
  double D_ZERO = 0.0;
  char TRANS = 'T';

  double *Vm = V + basis_size * m;
  double *VAm = VA + basis_size * m;

  dgemv_(&TRANS, &basis_size, &m, &D_ONE, V, &basis_size, VAm, &I_ONE, &D_ZERO, 
      submatrix + m*max_vectors, &I_ONE);
  submatrix[m*max_vectors + m] = ddot_(&basis_size, Vm, &I_ONE, VAm, &I_ONE);
}

static double calculate_residue(double* residue, double *result, double* eigv, double theta, 
    double* V, double* VA, int basis_size, int m){
  double D_ZERO = 0;
  double D_ONE = 1;
  int I_ONE = 1;
  char NTRANS = 'N';

  double mtheta = -theta;
  dgemv_(&NTRANS, &basis_size, &m, &D_ONE, V, &basis_size, eigv, &I_ONE, &D_ZERO, result, &I_ONE);
  dgemv_(&NTRANS,&basis_size,&m, &D_ONE, VA, &basis_size, eigv, &I_ONE, &D_ZERO, residue, &I_ONE);
  daxpy_(&basis_size, &mtheta, result, &I_ONE, residue, &I_ONE);
  return dnrm2_(&basis_size, residue, &I_ONE);
}

static void create_new_vec_t(double* residue, double* diagonal, double theta, int size){
  /* quick implementation */
  int i;
  double cutoff = 1e-12;
  for( i = 0 ; i < size ; i++)
    if(fabs(diagonal[i] - theta) > cutoff)
      residue[i] = residue[i] / fabs(diagonal[i] - theta);
    else{
      residue[i] = residue[i] / cutoff;
    }
}

static void deflate(double* sub_matrix, double* V, double* VA, int max_vectors, int basis_size, 
    int keep_deflate, double* eigv){

  double D_ZERO = 0;
  double D_ONE = 1;
  int I_ONE = 1;
  char NTRANS = 'N';
  int i;

  int size_x_deflate = basis_size * keep_deflate;

  double *new_result = safe_malloc(basis_size * keep_deflate, double);

  dgemm_(&NTRANS, &NTRANS, &basis_size, &keep_deflate, &max_vectors, &D_ONE, V, &basis_size, eigv, \
          &max_vectors, &D_ZERO, new_result , &basis_size);
  dcopy_(&size_x_deflate, new_result, &I_ONE, V, &I_ONE);

  /*
  for(i = 0 ; i < keep_deflate; i ++)
    matvec(V + i * basis_size, VA + i * basis_size); *//* only here expensive matvec needed */
  dgemm_(&NTRANS, &NTRANS, &basis_size, &keep_deflate, &max_vectors, &D_ONE, VA, &basis_size, eigv,\
     &max_vectors, &D_ZERO, new_result , &basis_size);

  dcopy_(&size_x_deflate, new_result, &I_ONE, VA, &I_ONE);
 
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

  safe_free(new_result);

  for( i = 0 ; i < keep_deflate; i++)
    expand_submatrix(sub_matrix, V, VA, max_vectors, basis_size, i);
}
