#include <stdio.h>
#include <stdlib.h> 
#include <omp.h> 

#include "lapack.h"
#include "macros.h"
#include "debug.h"

static unsigned int N;             /**< Number of electron pairs. */
static unsigned int L;             /**< Number of orbitals. */
static unsigned int LminN;         /**< L - N **/
static unsigned int basis_size;    /**< Number of basis states. */
static unsigned int *to_bitstring; /**< Array for transformation of cnt number to bitstring. */
static int *Y;                     /**< Array that helps with conversion from bitstring to cnt number. */
static double **Viikk;             /**< Two-particle integrals needed. */
static double core_energy;         /**< Core energy. */
static double* diagonal;           /**< Array with diagonal elements of the Hamiltonian. */
static int *occ_start;
static unsigned int *cnt_start;
static int threadnos;

#include "doci.h"
#include "read_fcidump.h"

#define CHAR_BIT 8

/* ============================================================================================== */
/* ================================ DECLARATION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

/* -------- INIT --------- */

static void init_Y1(void);

static unsigned int* init_x(void);

static void init_to_bitstring(void);

static void init_diagonal(double* one_p_int, double* two_p_int);

static void transform_integrals(double* one_p_int, double* two_p_int);

static void init_Viikk(double* two_p_int);

/* -------- HEFF --------- */

static void fill_occupation(unsigned int bitstring, int* occ);

static int next_bitstring(int cnt, int *exc_cnt, int* occ, int* exc_occ, int* to_incr, int* augm, double** V);

static void increment_occ(int *occ, int *exc_occ);

/* -------- MISC --------- */

static unsigned int combination(unsigned int n, unsigned int m);

/* ============================================================================================== */

void fill_doci_data(char* dumpfil){
  unsigned int Na, Nb;
  double *one_p_int, *two_p_int;
  read_fcidump(dumpfil, &L, &Na, &Nb, &core_energy, &one_p_int, &two_p_int);

  if( Na != Nb ) {
    fprintf(stderr, "Number of up and down electrons not equal in FCIDUMP-file.\n");
    fprintf(stderr, "This is obligatory for DOCI.\n");
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }
  N = Na;
  LminN = L - N;
  if ( CHAR_BIT *  sizeof( unsigned int ) < L) {
    fprintf(stderr, "Too many orbitals.\n");
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }

  /* transform integrals to appropriate FCI integrals */
  transform_integrals(one_p_int, two_p_int);

  basis_size = combination(N, L);

  #pragma omp parallel default(none) shared(threadnos)
  {
    threadnos = omp_get_num_threads();
  }

  init_Y1();
  init_to_bitstring();
  init_diagonal(one_p_int, two_p_int);
  init_Viikk(two_p_int);
  safe_free(one_p_int);
  safe_free(two_p_int);

  {
    int i;
    occ_start = safe_malloc((N + 1) * threadnos, int);
    cnt_start = safe_malloc(threadnos + 1, unsigned int);
    for( i = 0 ; i < threadnos ; i++ ){
      cnt_start[i] = i * basis_size / threadnos;
      fill_occupation(to_bitstring[cnt_start[i]], occ_start + ( N + 1 ) * i);
      occ_start[ N - 1 + ( N + 1 ) * i ]--;
    }
    cnt_start[i] = basis_size;
  }

  printf(
      "----------\n"
      "\t** executing DOCI (%d thread%s)\n"
      "\t** N = (%u,%u)\n"
      "\t** L = %u\n"
      "\t** basis size : %u\n"
      "----------\n",
      threadnos, threadnos == 1 ? "" : "s", N, N, L, basis_size
  );
}

double* doci_get_diagonal(void){return diagonal;}

int doci_get_basis_size(void){return basis_size;}

void doci_cleanup_data(void){
  safe_free(*Viikk);
  safe_free(Viikk);
  safe_free(Y);
  safe_free(to_bitstring);
  safe_free(diagonal);
  safe_free(cnt_start);
  safe_free(occ_start);
}

void doci_matvec(double* vec, double* result){
  
  double *result_xtemp = safe_calloc(basis_size * (threadnos - 1), double);

  #pragma omp parallel default(none) shared(result, vec, Viikk, to_bitstring, basis_size, N, L, \
      diagonal, threadnos, result_xtemp, occ_start, cnt_start)
  {
    unsigned int cnt;
    int occ[N + 1];
    int exc_occ[N + 1];
    const int thrno = omp_get_thread_num();
    double *result_x = thrno ? result_xtemp + basis_size * ( thrno - 1 ) : result;

    for(cnt = 0 ; cnt < N + 1 ; cnt++ ){
      occ[cnt] = occ_start[ cnt + (N + 1) * thrno ];
      exc_occ[cnt] = occ_start[ cnt + (N + 1) * thrno ];
    }

    #pragma omp for simd schedule(static)
    for( cnt = 0 ; cnt < basis_size ; cnt++ )
      result[cnt] = vec[cnt] * diagonal[cnt];

    for( cnt = cnt_start[thrno] ; cnt < cnt_start[thrno + 1] ; cnt++ ){
      int cnt_exc = cnt;
      int to_incr = N - 1;
      int augm = N - 1;
      double *V;
      increment_occ(occ, exc_occ);
      V = Viikk[occ[augm]];

      while( next_bitstring(cnt, &cnt_exc, occ, exc_occ, &to_incr, &augm, &V) ){
        result_x [ cnt ] += *V * vec [ cnt_exc ];
        result_x [ cnt_exc ] += *V * vec [ cnt ];
      }
    }
    /*
    printf("thread %d : %d counts\n", thrno, counter);
    */
    #pragma omp barrier

    #pragma omp for schedule(static)
    for( cnt = 0 ; cnt < basis_size ; cnt++ ){
      int i;
      for(i = 0 ; i < threadnos - 1 ; i++ )
        result[cnt] += result_xtemp[cnt + i * basis_size ];
    }
  }
  free(result_xtemp);
}

/* ============================================================================================== */
/* ================================= DEFINITION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

/* -------- INIT --------- */

static void init_Y1(void){
  int i;
  int NminL1 = L - N + 1;
  int ind = 0;
  unsigned int *x = init_x();
  unsigned int el;
  
  Y = safe_calloc(N * LminN + L, int);

  for(el = 0 ; el < N ; el++ ){
    unsigned int *x_temp = x + NminL1 * el + NminL1;

    int orb;
    for(orb = 0; orb < NminL1 ; orb++){
      Y[ ind ] = 0;
      for( i = 1; i <= orb; i++ )
        Y[ ind ] += x_temp[ i ];
      ind++;
    }
  }
  safe_free(x);

  for( i = N * LminN + L - 1 ; i > 0 ; i-- )
    Y[i] = Y[i] - Y[i - 1];
}

static unsigned int* init_x(void){
  int el;
  int NminL = L - N + 1;
  unsigned int *x = safe_calloc(( N + 1 ) * NminL, unsigned int);

  x[ ( N + 1 ) * NminL - 1 ] = 1;
  for(el = N ; el >= 0; el--){
    int orb;
    for(orb = NminL - 1 ; orb >= 0; orb--){
      if( el != N)
        x[ el * NminL + orb ] += x[ (el + 1) * NminL + orb ];
      if( orb != NminL - 1 )
        x[ el * NminL + orb ] += x[ el * NminL + orb + 1 ];
    }
  }
  return x;
}

static void init_to_bitstring(void){
  unsigned int pow_of_two[L];
  unsigned int i;
  unsigned int bitstring;
  int occupied[ N + 1 ];

  /* checking if the bitstrings can be stored as an unsigned int */
  assert ( CHAR_BIT * sizeof(unsigned int) >= L );

  /* calculating powers of two */
  pow_of_two[0] = 1;
  for( i = 1 ; i < L ; i++) pow_of_two[ i ] = pow_of_two[ i - 1 ] * 2;

  /* initializing bitstring */
  to_bitstring = safe_malloc(basis_size, unsigned int);
  
  /* current occupation of the electrons */
  for( i = 0 ; i < N ; i++ ) occupied[ i ] = i;
  occupied[N] = L;

  /* bitstring associated with the current occupation */
  bitstring = 0;
  for( i = 0 ; i < N ; i++ ) bitstring += pow_of_two[occupied[i]];
  to_bitstring[0] = bitstring;

  /* creating the bitstring */
  for( i = 1 ; i < basis_size ; i++ ){
    int j;
    unsigned int k;
    for( j = N - 1 ; j >= 0 ; j--){
      if( occupied[ j ] + 1 != occupied[ j + 1 ] ){
        occupied[ j ]++;
        break;
      }
      else if(occupied[ j - 1 ] + 1 != occupied[j]){
        occupied[ j - 1 ]++;
        for( k = j ; k < N ; k++ )
          occupied[ k ] = occupied[ k - 1 ] + 1;
        break;
      }
    }
    bitstring = 0;
    for( k = 0 ; k < N ; k++ ) bitstring += pow_of_two[occupied[k]];
    to_bitstring[i] = bitstring;
  }
}

static void init_diagonal(double* one_p_int, double* two_p_int){
  const int L2 = L * L;
  const int L3 = L2 * L;

  diagonal = safe_malloc(basis_size, double); 

  #pragma omp parallel default(none) shared(diagonal, core_energy, to_bitstring, L, N, basis_size, \
      one_p_int, two_p_int, threadnos)
  {
    unsigned int cnt = omp_get_thread_num() * basis_size / threadnos;
    unsigned int cnt_end = (omp_get_thread_num() + 1) * basis_size / threadnos;
    int bit[N + 1], bitdummy[N + 1];
    fill_occupation(to_bitstring[cnt], bit);
    bit[N - 1]--;
    for(;  cnt < cnt_end; cnt++ ){
      unsigned int i;
      double temp = core_energy;
      increment_occ(bit, bitdummy);

      for(i = 0 ; i < N; i++ ){
        int orb1 = bit[i];
        unsigned int orb2;
        unsigned int cnt_orb2 = 0;

        temp += 2 * one_p_int[ orb1 * ( 1 + L ) ];
        for( orb2 = 0 ; orb2 < L ; orb2++ ){
          int is_occ = bit[ cnt_orb2 ] == orb2;
          cnt_orb2 += is_occ;

          /* orb1 is occupied and orb2 is occupied */
          temp += 2 * is_occ * two_p_int[ orb1 * ( 1 + L ) + orb2 * ( L2 + L3 ) ];

          /* orb1 is occupied and orb2 is virtual */
          temp += (1 - is_occ) * two_p_int[orb1 * ( 1 + L3 ) + orb2 * ( L + L2 )];
        }
      }
      diagonal[cnt] = temp;
    }
  }
}

static void transform_integrals(double* one_p_int, double* two_p_int){
  unsigned int i, j, k ,l;
  unsigned int L2 = L * L;
  unsigned int L3 = L2 * L;

  for( i = 0 ; i < L ; i++ )
    for( j = 0 ; j <= i; j++ )
      one_p_int[i * L + j] = one_p_int[j * L + i];

  for( i = 0 ; i < L ; i++ )
    for( j = 0 ; j <= i; j++ )
      for( k = 0 ; k <= i; k++ )
        for( l = 0 ; l <= k; l++ ){
          int curr_ind = i + L * j + L2 * k + L3 * l;
          if( two_p_int[curr_ind] != 0 ){
            two_p_int[k + L * l + L2 * i + L3 * j] = two_p_int[curr_ind];
            two_p_int[j + L * i + L2 * l + L3 * k] = two_p_int[curr_ind];
            two_p_int[l + L * k + L2 * j + L3 * i] = two_p_int[curr_ind];
            two_p_int[j + L * i + L2 * k + L3 * l] = two_p_int[curr_ind];
            two_p_int[l + L * k + L2 * i + L3 * j] = two_p_int[curr_ind];
            two_p_int[i + L * j + L2 * l + L3 * k] = two_p_int[curr_ind];
            two_p_int[k + L * l + L2 * j + L3 * i] = two_p_int[curr_ind];
          }
        }

  for( k = 0 ; k < L ; k++ )
    for( l = 0 ; l < L ; l++ ){
      double temp = 0;
      for( j = 0 ; j < L ; j++ )
        temp +=  two_p_int[k + j*L + j*L2 + l*L3];
      one_p_int[k + l*L] += -0.5 * temp;
    }
}

static void init_Viikk(double* two_p_int){
  unsigned int i, k;
  int L2 = L * L + 1;
  int L3 = L2 * L;
  double* tempViikk = safe_malloc(L * ( L - 1 ) / 2 + 1, double);
  /* only upper triangle is stored */
  Viikk = safe_malloc(L, double*);
  tempViikk++;
  for( i = 0 ; i < L ; i++ ){
    Viikk[i] = tempViikk-1;
    for( k = i + 1 ; k < L ; k++ )
      *tempViikk++ = two_p_int[i * L2 + k * L3];

  }
}

/* -------- MISC --------- */

static unsigned int combination(unsigned int n, unsigned int m){
  unsigned long long r;
  unsigned int d;
  if( n > m )
    return 0;
  r = 1;
  for(d = 1 ; d <= n ; ++d ){
    r *= m--;
    r /= d;
  }
  return (unsigned int) r;
}

/* -------- HEFF --------- */

static void fill_occupation(unsigned int bitstring, int* occ){
  unsigned int orb;
  int cnt = 0;
  for(orb = 0; orb < L ; orb++){
    occ[cnt] = orb;
    cnt += bitstring % 2;
    bitstring = bitstring / 2;
  }
  occ[N] = L;
  assert(bitstring == 0);
}

static int next_bitstring(int cnt, int *exc_cnt, int* occ, int* exc_occ, int* to_incr, int* augm, double** V){
  unsigned int i;
  exc_occ[*to_incr]++;
  (*V)++;
  *exc_cnt += Y[*to_incr * LminN + exc_occ[*to_incr]];

  while( exc_occ[*to_incr] == exc_occ[*to_incr + 1] && exc_occ[*to_incr] != L ){
    int temp = exc_occ[*to_incr];
    exc_occ[*to_incr] = exc_occ[++*to_incr];
    exc_occ[*to_incr] = temp + 1;
    (*V)++;
    *exc_cnt += Y[*to_incr * LminN + exc_occ[*to_incr]];
  }

  if( exc_occ[*to_incr] == L ){
    if(*augm == 0)
      return 0;

    for( i = *augm ; i < N ; i++ ) exc_occ[i] = occ[i];
    *to_incr = --(*augm);
    assert(occ[*augm] * (L+1) < L * L );
    *V = Viikk[occ[*augm]];
    *exc_cnt = cnt;
    return next_bitstring(cnt, exc_cnt, occ, exc_occ, to_incr, augm, V);
  }
  return 1;
}

static void increment_occ(int *occ, int *exc_occ){
  unsigned int to_incr = N - 1;
  while( occ[ to_incr ]++ - to_incr == LminN ){
    to_incr--;
  }
  for( ; to_incr < N - 1 ; to_incr++ )
    occ[to_incr + 1] = occ[to_incr] + 1;
  for( to_incr = 0 ; to_incr < N ; to_incr++ )
    exc_occ[to_incr] = occ[to_incr];
}
