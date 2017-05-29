#include <stdio.h>
#include <stdlib.h> 
#include <omp.h> 
#include <time.h>

#include "lapack.h"
#include "debug.h"

/**
 * \brief A struct that wraps all the data needed for a matvec product.
 */
typedef struct matvec_data{
  int N;            /**< Number of electron pairs. */
  int ORBS;         /**< Number of orbitals. */
  int basis_size;   /**< Number of basis states. */
  unsigned int *to_bitstring; /**< Array for transformation of cnt number to bitstring. */
  unsigned int *Y; /**< Array that helps with conversion from bitstring to cnt number. */
  double *one_p_int; /**< Single-particle integrals. */
  double *two_p_int; /**< Two-particle integrals. */
  double core_energy; /**< Core energy. */
  double* diagonal; /**< Array with diagonal elements of the Hamiltonian. */
} matvec_data;

#include "DOCI.h"
#include "read_fcidump.h"
#include "wrapper_solvers.h"

#define CHAR_BIT 8

int main(int argc, char* argv[] ){
  char* filename;
  matvec_data data;
  double energy, *result, norm, tol;
  int ONE, cnt, max_its, david_keep, david_max_vec, HF_init;
  char dumpfil[255], solver[10];

  ONE = 1;

  if(argc != 2){
    printf("Illegal number of arguments, one argument (FCIDUMP-file) should be provided!\n");
    return EXIT_FAILURE;
  }

  filename = argv[1];
  read_inputfile(filename, dumpfil, solver, &max_its, &tol, &david_keep, &david_max_vec, &HF_init);
  read_fcidump(dumpfil, &(data.ORBS), &(data.N), &(data.N), &(data.core_energy), \
      &(data.one_p_int), &(data.two_p_int));

  if ( CHAR_BIT * sizeof(unsigned int) < data.ORBS) {
    printf("NO SENPAI, IT IS TOO BIG!\n");
    return EXIT_FAILURE;
  }

  data.basis_size = (int) combination(data.N, data.ORBS);

  printf("basissize : %d\n", data.basis_size);

  data.Y = initialize_Y1(data.N, data.ORBS);

  data.to_bitstring = initialize_to_bitstring(data.N, data.ORBS);

  /* transform integrals to appropriate FCI integrals */
  transform_integrals(data.one_p_int, data.two_p_int, data.ORBS);

  data.diagonal = create_diagonal(data.to_bitstring, data.basis_size, data.ORBS, data.one_p_int, \
      data.two_p_int, data.core_energy);

  printf("\nHartree Fock energy is : %f\n", data.diagonal[0]);
  result = (double*) calloc(sizeof(double), data.basis_size);
  if(HF_init)
    result[0] = 1;
  else{
    srand(time(NULL));
    for(cnt = 0 ; cnt < data.basis_size; cnt++ ) result[cnt] = rand();
    norm = 1/dnrm2_(&data.basis_size, result, &ONE);
    dscal_(&data.basis_size, &norm, result, &ONE);
  }

  printf("\n** EXECUTING DOCI CALCULATION **\n");

  sparse_eigensolve(result, data.basis_size, &energy, matvec, &data, data.diagonal, NULL, tol, \
      max_its, solver, david_keep, david_max_vec);

  free(data.Y);
  free(data.to_bitstring);
  free(data.diagonal);
  free(data.one_p_int);
  free(data.two_p_int);
  free(result);

  printf("converged energy : %.10lf\n", energy);
  return EXIT_SUCCESS;

}

/* DEBUGGING FUNCTIONS */
#ifdef DEBUG
void print_bitstrings(unsigned int* bitstrings, const unsigned int size, const int L){
  int i;
  unsigned int bitstring;
  DPRINTFUNC
  
  for ( i = 0 ; i < size ; i ++){
    bitstring = bitstrings[i];
    DPRINT("%d: %d\t: ", i, bitstring);
    print_bitstring(bitstring, L);
    DPRINT("\n");
    }

}

void print_bitstring(unsigned int bitstring, const int L){ int j, rest;
  for( j = 0 ; j < L ; j ++){
    rest = bitstring % 2;
    bitstring = (bitstring - rest) / 2;
    DPRINT("%d", rest);
  }
}

void print_Y(unsigned int* Y, const int N, const int NminL){
  int i,j;
  DPRINTFUNC;
  DPRINT("===== Y =====\n");
  for(i = 0; i < N ; i++){
    for(j = 0; j < NminL ; j++)
      DPRINT("%d ", Y[i*NminL + j]);
    DPRINT("\n");
  }
  DPRINT("=============\n");
}

void check_bitstrings(unsigned int* bitstrings, unsigned int* Y, const int N, const int L){
  unsigned int size, i, result, bitstring;
  
  DPRINTFUNC
  size = combination(N,L);
  for( i = 0 ; i < size ; i ++ ){
    bitstring = bitstrings[i];
    result = bitstring_to_cnt(bitstring, Y, N, L);
    if ( result != i ){
      DPRINT("Bitstrings incorrect created, bitstring nr. %u is wrong (value: %u).\n", i, \
          bitstrings[i]);
      DPRINT("Should be at nr. %u\n", result);
      break;
    }
  }
  DPRINT("All bitstrings are correct, congratz Klaas!\n");
}

void check_next_bitstring(unsigned int *to_bitstrings, unsigned int *Y, int N, int L){
  int i,j,sgn,cnt, basis_size, dummy_cnt, *occ,  *occ_exc, cnt_c, i_new, j_new;
  unsigned int exc_bitstring, bitstring;
  DPRINTFUNC;
  basis_size = (int) combination(N,L);
  occ = (int*) malloc(sizeof(int) * L);
  occ_exc = (int*) malloc(sizeof(int) * L);
  for(cnt = 0; cnt < basis_size ; cnt++){
    DPRINT("=============\n");
    bitstring = to_bitstrings[cnt];
    print_bitstring(bitstring, L);
    DPRINT("\n");
    fill_occupation(bitstring, occ, L);
    for( cnt_c = 0 ; cnt_c < L ; cnt_c++ ) occ_exc[cnt_c] = occ[cnt_c];
    i = -1;
    j = -1;
    dummy_cnt = cnt;
    while( (sgn = next_bitstring(cnt, &dummy_cnt,  &i, &j, occ, occ_exc, Y, L, N))){
      exc_bitstring = to_bitstrings[dummy_cnt];
      print_bitstring(exc_bitstring, L);
      i_new = i < j? i : j;
      j_new = i < j? j : i;
      DPRINT(" %d %d %d %d\n", dummy_cnt, sgn, i_new, j_new);
    }
  }
}
#else
void print_bitstrings(unsigned int* bitstrings, const unsigned int size, const int L){}
void print_bitstring(unsigned int bitstring, const int L){}
void print_Y(unsigned int* Y, const int N, const int NminL){}
void check_bitstrings(unsigned int* bitstrings, unsigned int* Y, const int N, const int L){}
void check_next_bitstring(unsigned int *to_bitstrings, unsigned int *Y, int N, int L){}
#endif
/* END OF DEBUGGING FUNCTIONS */

unsigned int* initialize_Y1(const int N, const int L){
  int el, orb, i, NminL;
  int index;
  unsigned int *x, *result, *x_temp;
  
  NminL = L - N + 1;
  result = ( unsigned int * ) malloc(sizeof(unsigned int) * N * NminL);
  x = initialize_x(N, L);

  index = 0;
  for(el = 0 ; el < N ; el++ ){
    x_temp = x + NminL * el + NminL;

    for(orb = 0; orb < NminL ; orb++){
      result[ index ] = 0;
      for( i = 1; i <= orb; i++ )
        result[ index ] += *(x_temp + i);
      index++;
    }
  }
  free(x);
  
  return result;
}

unsigned int* initialize_x(const int N, const int L){
  int el, orb, NminL;
  unsigned int *x;

  NminL = L - N + 1;
  x = (unsigned int*) calloc((N+1) * NminL, sizeof(unsigned int));

  x [ (N+1) * NminL - 1 ] = 1;
  for(el = N ; el >= 0; el--){
    for(orb = NminL - 1 ; orb >= 0; orb--){
      if( el != N)
        x[ el * NminL + orb ] += x[ (el + 1) * NminL + orb ];
      if( orb != NminL - 1 )
        x[ el * NminL + orb ] += x[ el * NminL + orb + 1 ];
    }
  }
  return x;
}

unsigned int combination(int N, int L){
  unsigned long long r;
  unsigned int d;
  if( N > L )
    return 0;
  r = 1;
  for(d = 1 ; d <= N ; ++d ){
    r *= L--;
    r /= d;
  }
  return (unsigned int) r;
}

unsigned int* initialize_to_bitstring(const int N, const int L){
  unsigned int *to_bitstring, *pow_of_two;
  unsigned int basis_size, bitstring, i, j, k;
  int* occupied;

  /* checking if the bitstrings can be stored as an unsigned int */
  assert ( CHAR_BIT * sizeof(unsigned int) >= L);

  /* calculating powers of two */
  pow_of_two = ( unsigned int * ) malloc(L * sizeof(unsigned int));
  pow_of_two[0] = 1;
  for( i = 1 ; i < L ; i++) pow_of_two[i] = pow_of_two[i-1] * 2;

  /* initializing bitstring */
  basis_size = combination(N, L);
  to_bitstring = ( unsigned int * ) malloc( basis_size * sizeof(unsigned int));
  
  /* current occupation of the electrons */
  occupied = (int*) malloc((N + 1) * sizeof(int));
  for( i = 0 ; i < N ; i++ ) occupied[i] = i;
  occupied[N] = L;

  /* bitstring associated with the current occupation */
  bitstring = 0;
  for( i = 0 ; i < N ; i++ ) bitstring += pow_of_two[occupied[i]];
  to_bitstring[0] = bitstring;

  /* creating the bitstring */
  for( i = 1; i < basis_size; i++ ){
    for( j = N - 1 ; j >= 0 ; j--){
      if( occupied[j] + 1 != occupied[j+1]){
        occupied[j]++;
        break;
      }
      else if(occupied[j - 1] + 1 != occupied[j]){
        occupied[j - 1]++;
        for( k = j ; k < N ; k++ )
          occupied[k] = occupied[k-1] + 1;
        break;
      }
    }
    bitstring = 0;
    for( j = 0 ; j < N ; j++ ) bitstring += pow_of_two[occupied[j]];
    to_bitstring[i] = bitstring;

  }

  free(occupied);
  free(pow_of_two);

  return to_bitstring;
}

unsigned int bitstring_to_cnt(unsigned int bitstring, unsigned int *Y, const int N, const int L){
  unsigned int N_el, NminL, rest, result, i;

  N_el = 0;
  NminL = L - N + 1;
  result = 0;

  for( i = 0 ; i < L ; i ++){
    rest = bitstring % 2;
    bitstring = (bitstring - rest) / 2;
    if(rest){
      result += Y[N_el * NminL + i - N_el];
      N_el++;
    }
  }
  return result;
}

void calculate_sigma_three(double* vec, double* two_p_int, unsigned int* to_bitstring, unsigned \
                            int *Y, int basis_size, const int N, const int L, double* result){
  int cnt, cnt_exc, k, l;
  int L3, L2, *occ, *exc_occ, cnt_c;
  unsigned int bitstring;

  L2 = L * L;
  L3 = L2 * L;

  #pragma omp parallel default(none) shared(result, vec, two_p_int,  to_bitstring, Y, basis_size, \
      L2, L3) \
    private(cnt, cnt_exc, k, l, bitstring, occ, exc_occ, cnt_c)
  {
    occ = (int*) malloc(L * sizeof(int));
    exc_occ = (int*) malloc(L * sizeof(int));

    #pragma omp for schedule(static)
    for( cnt = 0 ; cnt < basis_size ; cnt++ ){
      bitstring = to_bitstring[cnt];
      fill_occupation(bitstring, occ, L);
      for( cnt_c = 0 ; cnt_c < L ; cnt_c++ ) exc_occ[cnt_c] = occ[cnt_c];
      cnt_exc = cnt;
      k = -1;
      l = -1;
      while( (next_bitstring(cnt, &cnt_exc,  &k, &l, occ, exc_occ, Y, L, N)))
        result[ cnt ] += two_p_int[k + l*L + k*L2 + l*L3 ] * vec[ cnt_exc ];

    }
    free(occ);
    free(exc_occ);
  }
}

int next_bitstring(int cnt, int *exc_cnt, int *i, int *j, int* occ, int* exc_occ, \
    unsigned int *Y, const int L, const int N){

  int it_one, it_two;

  if(*exc_cnt == cnt){
    *i = -1;
    *j = -1;
    for(it_one = 0; it_one < L ; it_one++)
      if(occ[it_one] == 0){
        *i = it_one;
        break;
      }
    for(it_one = L -1; it_one >= 0 ; it_one--)
      if(occ[it_one]){
        *j = it_one;
        break;
      }
    if(*i == -1 || *j == -1)
      return 0;

    exc_occ[*i] = 1;
    exc_occ[*j] = 0;
    *exc_cnt = occ_to_cnt(exc_occ, Y, N, L);
    return 1;
  }

  exc_occ[*i] = 0;
  for(it_one = *i+1; it_one < L ; it_one++)
    if(occ[it_one] == 0){
      *i = it_one;
      exc_occ[*i] = 1;
      *exc_cnt = occ_to_cnt(exc_occ, Y, N, L);
      return 1;
    }

  exc_occ[*j] = 1;
  for(it_one = *j-1; it_one >= 0 ; it_one--)
    if(occ[it_one]){
      *j = it_one;
      exc_occ[*j] = 0;
      for(it_two = 0; it_two < L ; it_two++)
        if(occ[it_two] == 0){
          *i = it_two;
          exc_occ[*i] = 1;
          *exc_cnt = occ_to_cnt(exc_occ, Y, N, L);
          return 1;
        }
    }

  return 0;
}

int occ_to_cnt(int* occ, unsigned int* Y, const int N, const int L){
  int N_el, NminL, result, i;
  NminL = L - N;
  result = 0;
  N_el = 0;

  /**
   * Faster than if(occ[i]) result += Y[N_el + i]
   * to be expected due to pipelining i guess?
   */
  for( i = 0 ; i < L; i++){
    result += occ[i]*Y[N_el + i];
    N_el += occ[i] * NminL;
  }
return result;
}

void matvec(double* vec, double* result, void* vdata){
  int basis_size, cnt;

  matvec_data* data = (matvec_data*) vdata;

  basis_size = data->basis_size;
  for( cnt = 0 ; cnt < basis_size ; cnt++ ) result[cnt] = vec[cnt] * data->diagonal[cnt];

  calculate_sigma_three(vec, data->two_p_int, data->to_bitstring, data->Y, data->basis_size, \
      data->N, data->ORBS, result);
}

void transform_integrals(double* one_p_int, double* two_p_int, int ORBS){
  int i,j,k,l, ORBS2, ORBS3, curr_ind;
  double temp;
  ORBS2 = ORBS * ORBS;
  ORBS3 = ORBS2 * ORBS;

  for( i = 0 ; i < ORBS ; i++ )
    for( j = 0 ; j <= i; j++ )
      one_p_int[i * ORBS + j] = one_p_int[j * ORBS + i];

  for( i = 0 ; i < ORBS ; i++ )
    for( j = 0 ; j <= i; j++ )
      for( k = 0 ; k <= i; k++ )
        for( l = 0 ; l <= k; l++ ){
          curr_ind = i + ORBS * j + ORBS2 * k + ORBS3 * l;
          if(two_p_int[curr_ind] != 0){
          two_p_int[k + ORBS * l + ORBS2 * i + ORBS3 * j] = two_p_int[curr_ind];
          two_p_int[j + ORBS * i + ORBS2 * l + ORBS3 * k] = two_p_int[curr_ind];
          two_p_int[l + ORBS * k + ORBS2 * j + ORBS3 * i] = two_p_int[curr_ind];
          two_p_int[j + ORBS * i + ORBS2 * k + ORBS3 * l] = two_p_int[curr_ind];
          two_p_int[l + ORBS * k + ORBS2 * i + ORBS3 * j] = two_p_int[curr_ind];
          two_p_int[i + ORBS * j + ORBS2 * l + ORBS3 * k] = two_p_int[curr_ind];
          two_p_int[k + ORBS * l + ORBS2 * j + ORBS3 * i] = two_p_int[curr_ind];
          }
        }

  for( k = 0 ; k < ORBS ; k++ )
    for( l = 0 ; l < ORBS ; l++ ){
      temp = 0;
      for( j = 0 ; j < ORBS ; j++ )
        temp +=  two_p_int[k + j*ORBS + j*ORBS2 + l*ORBS3];
      one_p_int[k + l*ORBS] += -0.5 * temp;
    }
}

double* create_diagonal(unsigned int *to_bitstring, int basis_size, const int L, double *one_p_int,\
    double *two_p_int, double core_energy){

  double *diagonal, temp;
  int *bit, orb1, orb2, L2, L3, n_tot_orb1;
  unsigned int cnt;

  bit = (int*) malloc(L * sizeof(int)); diagonal = (double*) malloc(basis_size * sizeof(double)); 
  L2 = L*L;
  L3 = L2*L;

  for(cnt = 0 ;  cnt < basis_size ; cnt++ ){
    fill_occupation(to_bitstring[cnt], bit, L);
    temp = 0;

    for(orb1 = 0 ; orb1 < L; orb1++){
      n_tot_orb1 = 2*bit[orb1];
      temp += n_tot_orb1 * one_p_int[orb1 + orb1 * L];
      for(orb2 = 0 ; orb2 < L ; orb2++){
        temp += n_tot_orb1*bit[orb2] * two_p_int[orb1 + orb1*L + orb2*L2 + orb2*L3];

        /* orb1 is occupied and orb2 is virtual */
        temp += 0.5 * (n_tot_orb1 - 2* bit[orb1]*bit[orb2]) * two_p_int[orb1 + orb2 * L + orb2 * \
                L2 + orb1 * L3];
      }
    }
    diagonal[cnt] = temp + core_energy;
  }

  free(bit);
  return diagonal;
}

void fill_occupation(unsigned int bitstring, int* occ, const int ORBS){
  int orb, rest;
  for(orb = 0; orb < ORBS ; orb++){
    rest = bitstring % 2;
    bitstring = ( bitstring - rest )/2;
    occ[orb] = rest;
  }
  assert(bitstring == 0);
}
