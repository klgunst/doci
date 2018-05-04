#include <stdio.h>
#include <stdlib.h> 
#include <omp.h> 

#include "lapack.h"
#include "macros.h"
#include "debug.h"

struct doci_data
{
  unsigned int N;             /**< Number of electron pairs. */
  unsigned int L;             /**< Number of orbitals. */
  unsigned int basis_size;    /**< Number of Slater Determinants. */
  unsigned int *to_bitstring; /**< Array for transformation of cnt number to bitstring. */
  int *Y;                     /**< Array that helps with conversion from bitstring to cnt number. */
  double *Viikk;             /**< Two-particle integrals needed. */
  double core_energy;         /**< Core energy. */
  double* diagonal;           /**< Array with diagonal elements of the Hamiltonian. */
  int *occ_start;
  unsigned int *cnt_start;
  int threadnos;
}

#include "doci.h"
#include "read_fcidump.h"

#define CHAR_BIT 8

/* ============================================================================================== */
/* ================================ DECLARATION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

/* -------- INIT --------- */

static int* init_Y1( const unsigned int N, const unsigned int L );

static unsigned int* init_x( const unsigned int N, const unsigned int L );

static unsigned int* init_to_bitstring( const unsigned int N, const unsigned int L, 
    const unsigned int basis_size );

static double* init_diagonal( const double* const one_p_int, const double* const two_p_int, 
    struct doci_data *dat );

static void transform_integrals(double* one_p_int, double* two_p_int, const unsigned int L );
static double* init_Viikk(const double* const two_p_int, const unsigned int L );

/* -------- HEFF --------- */

static void fill_occupation( unsigned int bitstring, int occ[], const unsigned int N, const
    unsigned int L );

static const int next_bitstring( const unsigned int address, unsigned int *exc_address, 
    int occ[], int exc_occ[], int* to_incr, int* augm, double** V, const unsigned int L, const
    unsigned int N, const unsigned int LminN, const int *Y, double *Viikk, int **currpair );

static const int prev_bitstring( const unsigned int address, unsigned int *exc_address, 
    int occ[], int exc_occ[], int* to_incr, int* augm, double** V, const unsigned int L, const
    unsigned int N, const unsigned int LminN, const int *Y, double *Viikk, int **currpair );

static void increment_occ( int occ[], int exc_occ[], const unsigned int N, const unsigned int LminN );

/* -------- MISC --------- */

static unsigned int combination( unsigned int n, unsigned int m );

/* ============================================================================================== */

void fill_doci_data( char* dumpfil, void** vdat )
{
  unsigned int Na, Nb;
  double *one_p_int, *two_p_int;
  struct doci_data *result = safe_malloc( 1, struct doci_data );
  read_fcidump(dumpfil, &result->L, &Na, &Nb, &result->core_energy, &one_p_int, &two_p_int);

  if( Na != Nb )
  {
    fprintf(stderr, "Number of up and down electrons not equal in FCIDUMP-file.\n");
    fprintf(stderr, "This is obligatory for DOCI.\n");
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }
  result->N = Na;
  if ( CHAR_BIT *  sizeof( unsigned int ) < result->L )
  {
    fprintf(stderr, "Too many orbitals.\n");
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }

  /* transform integrals to appropriate FCI integrals */
  transform_integrals( one_p_int, two_p_int, result->L );

  result->basis_size = combination( result->N, result->L );

  #pragma omp parallel default(none) shared( result )
  {
    result->threadnos = omp_get_num_threads();
  }

  result->Y            = init_Y1( result->N, result->L );
  result->to_bitstring = init_to_bitstring( result->N, result->L, result->basis_size );
  result->diagonal     = init_diagonal( one_p_int, two_p_int, result );
  result->Viikk        = init_Viikk( two_p_int, result->L );
  safe_free(one_p_int);
  safe_free(two_p_int);

  {
    int i;
    result->occ_start = safe_malloc( result->N * result->threadnos, int);
    result->cnt_start = safe_malloc( result->threadnos + 1, unsigned int);
    for( i = 0 ; i < result->threadnos ; i++ )
    {
      result->cnt_start[i] = i * result->basis_size / result->threadnos;
      fill_occupation(result->to_bitstring[ result->cnt_start[ i ] ], 
          result->occ_start + result->N * i, result->N, result->L );
      result->occ_start[ result->N - 1 + result->N * i ]--;
    }
    result->cnt_start[ i ] = result->basis_size;
  }

  printf(
      "----------\n"
      "\t** executing DOCI (%d thread%s)\n"
      "\t** N = (%u,%u)\n"
      "\t** L = %u\n"
      "\t** basis size : %u\n"
      "----------\n",
      result->threadnos, result->threadnos == 1 ? "" : "s", result->N, result->N, result->L, 
      result->basis_size
  );

  *vdat = result;
}

double* doci_get_diagonal( void *vdat )
{ 
  struct doci_data *dat = vdat;
  return dat->diagonal; 
}

int doci_get_basis_size( void *vdat )
{ 
  struct doci_data *dat = vdat;
  return dat->basis_size; 
}

void doci_cleanup_data( void **vdat )
{
  struct doci_data *dat = *vdat;
  safe_free( (dat)->Viikk );
  safe_free( (dat)->Y );
  safe_free( (dat)->to_bitstring );
  safe_free( (dat)->diagonal );
  safe_free( (dat)->cnt_start );
  safe_free( (dat)->occ_start );
  safe_free( dat );
  *vdat = NULL;
}

/* This only uses the lower triangular part of the hamiltonian, and this way, i need to take
 * into account the race condition when parallelisation
 */
void doci_matvec(double* vec, double* result, void *vdat )
{
  struct doci_data *dat = vdat;
  const unsigned int N = dat->N;
  const unsigned int L = dat->L;
  const unsigned int basis_size = dat->basis_size;
  const int Lp1 = L + 1;
  const unsigned int LminN = L - N;

  double *result_xtemp = safe_calloc( basis_size * (dat->threadnos - 1), double);

  #pragma omp parallel default(none) shared(result, vec, dat, result_xtemp )
  {
    unsigned int cnt;
    int occ[ N ];
    int exc_occ[ N ];
    const int thrno = omp_get_thread_num();
    double *result_x = thrno ? result_xtemp + basis_size * ( thrno - 1 ) : result;

    /* copy the first occupation array to occ and exc_occ */
    for(cnt = 0 ; cnt < N ; ++cnt )
    {
      occ[ cnt ]     = dat->occ_start[ cnt + N * thrno ];
      exc_occ[ cnt ] = dat->occ_start[ cnt + N * thrno ];
    }

    /* do in parallel the diagonal part of the hamiltonian */
    #pragma omp for simd schedule( static )
    for( cnt = 0 ; cnt < basis_size ; ++cnt )
      result[ cnt ] = vec[ cnt ] * dat->diagonal[ cnt ];

    /* Do for each thread all exciations for SD's ranging from cnt_start[thrno] to 
     * cnt_start[thrno + 1] */
    for( cnt = dat->cnt_start[ thrno ] ; cnt < dat->cnt_start[ thrno + 1 ] ; ++cnt )
    {
      unsigned int cnt_exc = cnt;
      /* Of the new SD try first to excite the last occupied orbital */
      int to_incr = N - 1; /* the index number of the pair that is excited in this 
                              excited SD is stored in here */
      int augm    = N - 1; /* the index number of the pair that is excited in the original SD
                              is stored in here */
      double *V;
      int *currpair = &exc_occ[ to_incr ];

      /* Go to next occupation array of next SD, for the first SD the effect of this fnction
       * is negated by how dat->occ_start is set */
      increment_occ( occ, exc_occ, N, LminN );

      /* Set pointer of interaction to Viiii with i set to last occupied orbital */
      V = &dat->Viikk[ occ[ augm ] * Lp1 ];

      /* calculate all the excitations of this SD */
      while( prev_bitstring(cnt, &cnt_exc, occ, exc_occ, &to_incr, &augm, &V, L, N, LminN, dat->Y, 
            dat->Viikk , &currpair ))
      {
        double value = *V;
        result_x[ cnt ]     += value * vec[ cnt_exc ];
        result_x[ cnt_exc ] += value * vec[ cnt ];
      }
    }

    /* Wait for all threads to end */
    #pragma omp barrier

    /* sum all the results from the different threads */
    #pragma omp for schedule(static)
    for( cnt = 0 ; cnt < basis_size ; cnt++ )
    {
      int i;
      for(i = 0 ; i < dat->threadnos - 1 ; ++i )
        result[ cnt ] += result_xtemp[cnt + i * basis_size ];
    }
  }
  free(result_xtemp);
}

/* This only uses the upper triangular part of the hamiltonian, and this way, i need to take
 * into account the race condition when parallelisation
 *
 * For some reason this is noticeably slower than the lower part???
 */
/*
void doci_upper_matvec(double* vec, double* result, void *vdat )
{
  struct doci_data *dat = vdat;
  const unsigned int N = dat->N;
  const unsigned int L = dat->L;
  const unsigned int basis_size = dat->basis_size;
  const int Lp1 = L + 1;
  const unsigned int LminN = L - N;

  double *result_xtemp = safe_calloc( basis_size * (dat->threadnos - 1), double);

  #pragma omp parallel default(none) shared(result, vec, dat, result_xtemp )
  {
    unsigned int cnt;
    int occ[ N ];
    int exc_occ[ N ];
    const int thrno = omp_get_thread_num();
    double *result_x = thrno ? result_xtemp + basis_size * ( thrno - 1 ) : result;

    for(cnt = 0 ; cnt < N ; ++cnt )
    {
      occ[ cnt ]     = dat->occ_start[ cnt + N * thrno ];
      exc_occ[ cnt ] = dat->occ_start[ cnt + N * thrno ];
    }

    #pragma omp for simd schedule( static )
    for( cnt = 0 ; cnt < basis_size ; ++cnt )
      result[ cnt ] = vec[ cnt ] * dat->diagonal[ cnt ];

    for( cnt = dat->cnt_start[ thrno ] ; cnt < dat->cnt_start[ thrno + 1 ] ; ++cnt )
    {
      unsigned int cnt_exc = cnt;
      int to_incr = N - 1; 
      int augm    = N - 1; 
      double *V;
      int *currpair = &exc_occ[ to_incr ];

      increment_occ( occ, exc_occ, N, LminN );

      V = &dat->Viikk[ occ[ augm ] * Lp1 ];

      while( next_bitstring(cnt, &cnt_exc, occ, exc_occ, &to_incr, &augm, &V, L, N, LminN, dat->Y, 
            dat->Viikk , &currpair ))
      {
        double value = *V;
        result_x[ cnt ]     += value * vec[ cnt_exc ];
        result_x[ cnt_exc ] += value * vec[ cnt ];
      }
    }

    #pragma omp barrier

    #pragma omp for schedule(static)
    for( cnt = 0 ; cnt < basis_size ; cnt++ )
    {
      int i;
      for(i = 0 ; i < dat->threadnos - 1 ; ++i )
        result[ cnt ] += result_xtemp[cnt + i * basis_size ];
    }
  }
  free(result_xtemp);
}
*/

/*
 * This calculates the full matvec and this way doesnt have a race condition in parallelisation
 * 
 */
/*
void doci_full_matvec(double* vec, double* result, void *vdat )
{
  struct doci_data *dat = vdat;
  const unsigned int N = dat->N;
  const unsigned int L = dat->L;
  const unsigned int basis_size = dat->basis_size;
  const int Lp1 = L + 1;
  const unsigned int LminN = L - N;

  #pragma omp parallel default(none) shared(result, vec, dat )
  {
    unsigned int cnt;
    int occ[ N ];
    int exc_occ[ N ];
    const int thrno = omp_get_thread_num();

    for(cnt = 0 ; cnt < N ; ++cnt )
    {
      occ[ cnt ]     = dat->occ_start[ cnt + N * thrno ];
      exc_occ[ cnt ] = dat->occ_start[ cnt + N * thrno ];
    }

    #pragma omp for simd schedule( static )
    for( cnt = 0 ; cnt < basis_size ; ++cnt )
      result[ cnt ] = vec[ cnt ] * dat->diagonal[ cnt ];

    for( cnt = dat->cnt_start[ thrno ] ; cnt < dat->cnt_start[ thrno + 1 ] ; ++cnt )
    {
      unsigned int cnt_exc = cnt;
      int to_incr = N - 1;
      int augm    = N - 1;
      double *V;
      int *currpair = &exc_occ[ to_incr ];
      int i;

      increment_occ( occ, exc_occ, N, LminN );

      V = &dat->Viikk[ occ[ augm ] * Lp1 ];

      while( prev_bitstring(cnt, &cnt_exc, occ, exc_occ, &to_incr, &augm, &V, L, N, LminN, dat->Y, 
            dat->Viikk , &currpair ))
      {
        double value = *V;
        result[ cnt ] += value * vec[ cnt_exc ];
      }
      to_incr = N - 1;
      augm    = N - 1;
      cnt_exc = cnt;
      currpair = &exc_occ[ to_incr ];
      for( i = 0 ; i < N ; ++i ) exc_occ[ i ] = occ[ i ];
      V = &dat->Viikk[ occ[ augm ] * Lp1 ];

      while( next_bitstring(cnt, &cnt_exc, occ, exc_occ, &to_incr, &augm, &V, L, N, LminN, dat->Y, 
            dat->Viikk , &currpair ))
      {
        double value = *V;
        result[ cnt ] += value * vec[ cnt_exc ];
      }
    }
  }
}
*/

/* ============================================================================================== */
/* ================================= DEFINITION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

/* -------- INIT --------- */

static int* init_Y1( const unsigned int N, const unsigned int L )
{
  /**
   * This function initializes the string graph that is used for mapping Slater determinants too
   * addresses.
   *
   * Y1 corresponds with lexical ordering and where the arcs Y1 are given weight. As corresponding 
   * with http://vergil.chemistry.gatech.edu/notes/ci.pdf at page 40 and further.
   *
   * Y1( e, o ) = x( e + 1, o + 1 ) + x ( e + 1, o ) + ... + x( e + 1, e + 1 )
   */
  int i;
  const int nr_rows = L - N + 1;
  int ind = 0;
  unsigned int *x = init_x( N, L ); // Initializes the weights of the vertices
  unsigned int e;
  
  /** REMARK : why + L? */
  int *Y = safe_calloc( N * ( L - N ) + L, int );

  for( e = 0 ; e < N ; ++e )
  {
    /* Pointer to the right column that i need of the vertex-weights */
    unsigned int *x_temp = &x[ ( e + 1 ) * nr_rows ];

    int o_min_e;
    for( o_min_e = 0; o_min_e < nr_rows ; ++o_min_e )
    {
      Y[ ind ] = 0;
      /** can be faster **/
      for( i = 1; i <= o_min_e; ++i )
      {
        Y[ ind ] += x_temp[ i ];
      }
      ++ind;
    }
  }
  safe_free( x );

  /** So at this moment I have the string graph as given in figure 1a page 41 of 
   *  http://vergil.chemistry.gatech.edu/notes/ci.pdf
   *
   *  The next step is where I subtract the previous arch value from every current arch value.
   *  This is so that i can later calculate a new address from a previous known address in one step 
   *  if the two SD's just differ in one place. See functions prev_bitstring and next_bitstring
   */
  for( i = N * ( L - N ) + L - 1 ; i > 0 ; --i )
    Y[ i ] -= Y[ i - 1 ];

  return Y;
}

static unsigned int* init_x( const unsigned int N, const unsigned int L )
{
  /** This function initializes the arc weights of the string graph in Lexical order.
   *  (see http://vergil.chemistry.gatech.edu/notes/ci.pdf for more information about Lexical 
   *  order).
   *
   *  x( e, o ) is stored in x[ e * ( L - N + 1 ) + ( o - e ) ]. 
   *  x( N, L ) = 1.
   *  x( e, o ) = x( e + 1, o + 1 ) + x( e, o + 1 )
   */
  int e;
  const int nr_rows = L - N + 1;
  unsigned int *x = safe_calloc( ( N + 1 ) * nr_rows, unsigned int );

  /**  x( N, L ) = 1. */
  x[ N * nr_rows + L - N ] = 1;

  for( e = N ; e >= 0; --e )
  {
    int o_min_e;
    for( o_min_e = L - N  ; o_min_e >= 0; --o_min_e )
    {
      /**  x( e, o ) = x( e + 1, o + 1 ) + x( e, o + 1 ) 
       *   and x( e, o ) = 0 if e > N or (o - e) > (N - L)
       *
       *   o - e = o_min_e
       */
      if( e != N )
        x[ e * nr_rows + o_min_e ] += x[ ( e + 1 ) * nr_rows + o_min_e ];
      if( o_min_e != L - N )
        x[ e * nr_rows + o_min_e ] += x[ e * nr_rows + ( o_min_e + 1 ) ];
    }
  }
  return x;
}

static unsigned int* init_to_bitstring( const unsigned int N, const unsigned int L, 
    const unsigned int basis_size )
{
  /** This function initializes the to_bitstring array. thus address of SD to bitstring.
   *  I don't need this really while calculating my matvec, I only use this for the start of every 
   *  thread to know the occupation array.
   */
  unsigned int pow_of_two[ L ];
  unsigned int i;
  unsigned int bitstring;
  unsigned int *to_bitstring;
  int occupied[ N + 1 ]; /** In this array the occupied orbitals are inputted. The array ends 
                           * with an extra value ( hence N + 1 ) of L ( number of orbitals )
                           */

  /* checking if the bitstrings can be stored as an unsigned int */
  assert ( CHAR_BIT * sizeof( unsigned int ) >= L );

  /* calculating powers of two */
  pow_of_two[ 0 ] = 1;
  for( i = 1 ; i < L ; ++i ) pow_of_two[ i ] = pow_of_two[ i - 1 ] * 2;

  /* initializing bitstring */
  to_bitstring = safe_malloc( basis_size, unsigned int );
  
  /* current occupation of the electrons, this is the occupation of the first slater determinant.
   * Thus: 0,1,2,3,4,5,... and closed by L!!! */
  for( i = 0 ; i < N ; ++i ) occupied[ i ] = i;
  occupied[ N ] = L;

  /* bitstring associated with the current occupation, thus atm just the first slater determinant.
   * Or 11111...0000000... */
  bitstring = 0;
  for( i = 0 ; i < N ; ++i ) bitstring += pow_of_two[ occupied[ i ] ];
  to_bitstring[ 0 ] = bitstring;

  /* creating the bitstring-array for the other slaters */
  for( i = 1 ; i < basis_size ; ++i )
  {
    int j;
    unsigned int k;
    /* I always first try to excite the last orbital, if that doesnt work, the second to last
     * and so on...
     */
    for( j = N - 1 ; j >= 0 ; --j )
    {
      /* So now i try to excite orbital particle j that is in orbital occupied[ j ] to the next
       * orbital ( thus orbital = occupied[ j ] + 1 ).
       *
       * However, it is possible that there is already another particle ( or pair for DOCI ) in this
       * orbital. If so, that will be pair number j + 1.
       *
       * This is why the occupied array is length N + 1, with as last element a value of L, this 
       * way i don't need an extra check to see if occupied[ j ] + 1 is out of bound ( i mean not 
       * larger than the number of actual orbitals I have )
       *
       * If particle j is successfully excited to the next orbital, all the higher particles
       * ( k > j ) are placed in order right after particle j.
       * e.g. if particle j is excited to orbital 5, particle j+1,j+2,j+3,... are in orb 6,7,8...
       */
      if( occupied[ j ] + 1 != occupied[ j + 1 ] )
      {
        occupied[ j ]++;
        for( k = j + 1 ; k < N ; k++ )
          occupied[ k ] = occupied[ k - 1 ] + 1;
        break;
      }
    }
    
    /* Transforms the occupied array to a bitstring */
    bitstring = 0;
    for( k = 0 ; k < N ; ++k ) bitstring += pow_of_two[ occupied[ k ] ];
    to_bitstring[ i ] = bitstring;
  }

  return to_bitstring;
}

static double* init_diagonal( const double* const one_p_int, const double* const two_p_int, 
    struct doci_data *dat )
{
  const unsigned int N = dat->N;
  const unsigned int L = dat->L;
  const unsigned int LminN = L - N;
  const unsigned int L2 = L * L;
  const unsigned int L3 = L2 * L;
  const unsigned int Lp1 = L + 1;
  const unsigned int L2pL3 = L2 + L3;
  const unsigned int onepL3 = 1 + L3;
  const unsigned int LpL2 = L + L2;

  double* const diagonal = safe_malloc( dat->basis_size, double ); 

  /** Here I parallelize over the different threads */
  #pragma omp parallel default(none) shared( dat )
  {
    /* definition of every the area of SD's where every thread should calculate over */
    /* So for every thread, cnt and cnt_end will be something else */
    unsigned int cnt           = omp_get_thread_num()   * dat->basis_size / dat->threadnos;
    unsigned int cnt_end = ( omp_get_thread_num() + 1 ) * dat->basis_size / dat->threadnos;

    /* Again definition of an occupation array, again the last element is L */
    /* occdummy is not needed for me here */
    int occ[ dat->N + 1 ], occdummy[ dat->N ];

    /* Fills the occupation for every bitstring */
    fill_occupation( dat->to_bitstring[ cnt ], occ, N, L );
    /* This decrement is needed, because I will call increment_occ in the start of every 
     * for-loop ( see few lines down ). But for the first SD I calculate the diagonal element for
     * no increment should be needed for the occupation because I already got that occupation
     * from fill_occupation. However, increment_occ will increment the occupation to the next SD
     * so to make sure to negate this first unwanted increment, I do a decrement on the occupation
     * of this first SD.
     */
    --occ[ N - 1 ];
    occ[ N ] = L;

    for(;  cnt < cnt_end; ++cnt )
    {
      int i;
      double temp = dat->core_energy;
      /* get the occupied orbitals of the next SD */
      increment_occ( occ, occdummy, N, LminN );

      /* loop over all occupied orbitals in SD */
      for(i = N - 1 ; i >= 0 ; --i )
      {
        const int orb1 = occ[ i ];
        unsigned int orb2;
        unsigned int cnt_orb2 = 0;
        unsigned int index1 = orb1 * Lp1;
        unsigned int index2 = orb1 * onepL3;

        /* Tii terms */
        temp += 2 * one_p_int[ index1 ];
        for( orb2 = 0 ; orb2 < L ; ++orb2 )
        {
          /* check if orb2 is occupied in the SD */
          const int is_occ = occ[ cnt_orb2 ] == orb2;
          cnt_orb2 += is_occ;

          /* orb1 is occupied and orb2 is occupied */
          /* Vijji terms */
          temp += 2 * is_occ * two_p_int[ index1 ];

          /* orb1 is occupied and orb2 is virtual */
          /* Viijj terms */
          temp += ( 1 - is_occ ) * two_p_int[ index2 ];
          index1 += L2pL3;
          index2 += LpL2;
        }
      }
      diagonal[ cnt ] = temp;
    }
  }
  return diagonal;
}

static void transform_integrals(double* one_p_int, double* two_p_int, const unsigned int L )
{
  unsigned int i, j, k ,l;
  const unsigned int L2 = L * L;
  const unsigned int L3 = L2 * L;

  for( i = 0 ; i < L ; ++i )
    for( j = 0 ; j <= i; ++j )
      one_p_int[ i * L + j ] = one_p_int[ j * L + i ];

  for( i = 0 ; i < L ; ++i )
    for( j = 0 ; j <= i; ++j )
      for( k = 0 ; k <= i; ++k )
        for( l = 0 ; l <= k; ++l ){
          int curr_ind = i + L * j + L2 * k + L3 * l;
          if( two_p_int[ curr_ind ] != 0 )
          {
            two_p_int[ k + L * l + L2 * i + L3 * j ] = two_p_int[ curr_ind ];
            two_p_int[ j + L * i + L2 * l + L3 * k ] = two_p_int[ curr_ind ];
            two_p_int[ l + L * k + L2 * j + L3 * i ] = two_p_int[ curr_ind ];
            two_p_int[ j + L * i + L2 * k + L3 * l ] = two_p_int[ curr_ind ];
            two_p_int[ l + L * k + L2 * i + L3 * j ] = two_p_int[ curr_ind ];
            two_p_int[ i + L * j + L2 * l + L3 * k ] = two_p_int[ curr_ind ];
            two_p_int[ k + L * l + L2 * j + L3 * i ] = two_p_int[ curr_ind ];
          }
        }

  for( k = 0 ; k < L ; k++ )
    for( l = 0 ; l < L ; l++ )
    {
      double temp = 0;
      for( j = 0 ; j < L ; j++ )
        temp +=  two_p_int[ k + j * L + j * L2 + l * L3 ];
      one_p_int[ k + l * L ] -= 0.5 * temp;
    }
}

static double* init_Viikk(const double* const two_p_int, const unsigned int L )
{
  unsigned int i, k;
  const unsigned int L2 = L * L + 1;
  const unsigned int L3 = L2 * L;
  double* Viikk = safe_malloc( L * L, double );
  double* Vcurr = Viikk;
  for( i = 0 ; i < L ; ++i )
  {
    for( k = 0 ; k < L ; ++k )
      if( i == k )
        *(Vcurr++) = 0;
      else
        *(Vcurr++) = two_p_int[ i * L2 + k * L3 ];
  }
  return Viikk;
}

/* -------- MISC --------- */

static unsigned int combination( unsigned int n, unsigned int m )
{
  /** Just regular combinatorics */
  unsigned long long r;
  unsigned int d;
  if( n > m )
    return 0;
  r = 1;
  for( d = 1 ; d <= n ; ++d )
  {
    r *= m--;
    r /= d;
  }
  return ( unsigned int ) r;
}

/* -------- HEFF --------- */

static void fill_occupation( unsigned int bitstring, int occ[], const unsigned int N, const
    unsigned int L )
{
  /** This function fills the occ-array for the bitstring given.
   *  The occ array is terminated with occ[ N ] = L
   */
  unsigned int orb;
  /* cnt keeps which particle pair your at */
  int cnt = 0;
  for( orb = 0; orb < L ; orb++ )
  {
    occ[ cnt ] = orb;
    /* if this certain orbital is filled, go to next pair (increment of cnt) */
    cnt += bitstring % 2;
    bitstring = bitstring / 2;
  }
  //occ[ N ] = L;
  assert( bitstring == 0 );
}

static const int next_bitstring( const unsigned int address, unsigned int *exc_address, 
    int occ[], int exc_occ[], int* to_incr, int* augm, double** V, const unsigned int L, const
    unsigned int N, const unsigned int LminN, const int *Y, double *Viikk, int **currpair )
{
  /** This calculates the address of the nextexcitation of the given SD, in some kind of
   * specific way. If you only call this function you only get the upper triangular contributions
   * of the Hamiltonian.
   *
   * \param address:     This is the address of the original SD.
   * \param exc_address: Here the address of the next excited SD is stored to.
   *                     Furthermore, the address of the previously calculated excited SD is 
   *                     passed through this. This because the address of the next excited SD
   *                     is fast calculatable by using the previous one.
   * \param occ:         The occupied orbitals of original SD.
   * \param exc_occ:     The occupied orbitals of the excited SD.
   * \param to_incr:     This is the position of the excited electron pair in the exc_occ array.
   * \param augm:        This is the position of the excited electron pair in the occ array.
   * \param V:           The pointer to the interaction term that was used for the previous 
   *                     excitation. Pointer to the interaction term for the next excitation is
   *                     passed through here.
   * \param L:           Nr of orbitals.
   * \param N:           Nr of pairs.
   * \param LminN:       L - N
   * \param Y:           Array that helps for address specification.
   * \param Viikk:       Interactions array.
   * \param currpair     This is equal too &exc_occ[ *to_incr ], thus pointer too the excited 
   *                     electron pair in the exc_occ array.
   *
   */

  /**
   * First the electron pair I previously excited, I will try to excite to another higher orbital.
   *
   * change exc_occ : increment one time the orbital of the electron pair.
   * change the exc_address appropriately.
   *          *exc_address += Y( e, o ) = Y[ e * ( L - N + 1 ) + o - e ]
   * change the needed interaction term : Viijj => Vii(j+1)(j+1)
   *
   * The pair is now excited from orbital i to orbital j+1 instead of to orbital j.
   */
  ++(*currpair)[ 0 ];
  *exc_address += Y[ *to_incr * LminN + (*currpair)[ 0 ] ];
  ++(*V);

  /* So now check if this excitation is valid.
   * Two possibilities why not valid:
   *    1) The pair I excited, ended up in an orbital that was already occupied.
   *       Since I increase the orbital number every time and the exc_occ array is sorted from
   *       low to high, this can only happen if exc_occ[ curr ] == exc_occ[ curr + 1 ].
   *
   *    2) I excited the pair too an orbital index too high, i.e. index=L
   *
   * If the first case is true, I do the thing in the while loop.
   * If the second case is true, i go to the if clause after this while loop, this second case can
   * only occur if *to_incr = N - 1
   *
   * If my excitation is OK i just return 1 and exit the function.
   */
  while( *to_incr != N - 1 && (*currpair)[ 0 ] == (*currpair)[ 1 ] )
  {
    /** So apparently you were in case 1:
     *  So my exc_occ looks like this : .. .. .. .. 5 5 .. .. .. for example.
     *  The succession of 5 and 5 is wrong ofcourse.
     *  The second 5 comes for example from 4 that I increased to 4.
     *  So a pair that comes from 4 cant be placed to 5, but maybe to 6, this I fix
     *  by increasing the *to_incr and increasing the exc_occ[ *to_incr ]
     *  So that i get .. .. .. .. 5 6 .. .. ..
     *
     *  So first i increase the to_incr.
     *  Then I reset the pointer currpair to the right exc_occ element.
     *  I increase the orbital of the electron pair.
     *  I change the exc_address appropriately.
     * change the needed interaction term : Viijj => Vii(j+1)(j+1)
     *
     * and after that recheck if this was a valid move. And make the right conclusion in my while
     * loop.
     */
    ++(*to_incr);
    *currpair = &exc_occ[ *to_incr ];
    ++(*currpair)[ 0 ];
    *exc_address += Y[ *to_incr * LminN + (*currpair)[ 0 ] ];
    ++(*V);
  }

  if( (*currpair)[ 0 ] == L )
  {
    /* So apparently i was in case 2:
     * This means that the pair that I excited from orbital *augm in the original SD cant be 
     * excited further.
     * So instead i try to excite the previous orbital and see if this works.
     */
    int i;
    /* If i am already exciting the first orbital and that failed, I cant do anything more
     * and i found all the excitations from this particular SD. */
    if( *augm == 0 )
      return 0;

    /* If it was not zero, I just reinitialise exc_occ again to occ. */
    for( i = *augm ; i < N ; ++i ) exc_occ[ i ] = occ[ i ];
    /* I decrease augm, and say this is also the pair im going to excite, meaning im going to
     * excite the previous pair in the SD */
    *to_incr = --( *augm );
    *currpair = &exc_occ[ *to_incr ];

    /* Put the pointer of the interaction to Viiii with i = occ[ *augm ] */
    *V = &Viikk[ occ[ *augm ] * L + occ[ *augm ] ];
    /* Put the exc_address to the original SD */
    *exc_address = address;

    /* do next_bitstring but this time, i am incrementing the doci pair excited from the new 
     * orbital */
    return next_bitstring( address, exc_address, occ, exc_occ, to_incr, augm, V, L, N, LminN, Y, 
        Viikk, currpair );
  }
  return 1;
}

static const int prev_bitstring( const unsigned int address, unsigned int *exc_address, 
    int occ[], int exc_occ[], int* to_incr, int* augm, double** V, const unsigned int L, const
    unsigned int N, const unsigned int LminN, const int *Y, double *Viikk, int **currpair )
{
  /** This calculates the address of the previous excitation of the given SD, in some kind of
   * specific way. If you only call this function you only get the lower triangular contributions
   * of the Hamiltonian.
   *
   * \param address:     This is the address of the original SD.
   * \param exc_address: Here the address of the next excited SD is stored to.
   *                     Furthermore, the address of the previously calculated excited SD is 
   *                     passed through this. This because the address of the next excited SD
   *                     is fast calculatable by using the previous one.
   * \param occ:         The occupied orbitals of original SD.
   * \param exc_occ:     The occupied orbitals of the excited SD.
   * \param to_incr:     This is the position of the excited electron pair in the exc_occ array.
   * \param augm:        This is the position of the excited electron pair in the occ array.
   * \param V:           The pointer to the interaction term that was used for the previous 
   *                     excitation. Pointer to the interaction term for the next excitation is
   *                     passed through here.
   * \param L:           Nr of orbitals.
   * \param N:           Nr of pairs.
   * \param LminN:       L - N
   * \param Y:           Array that helps for address specification.
   * \param Viikk:       Interactions array.
   * \param currpair     This is equal too &exc_occ[ *to_incr ], thus pointer too the excited 
   *                     electron pair in the exc_occ array.
   *
   */

  /**
   * First the electron pair I previously excited, I will try to excite to another lower orbital.
   *
   * change the exc_address appropriately.
   *          *exc_address -= Y( e, o ) = Y[ e * ( L - N + 1 ) + o - e ]
   * change exc_occ : decrement one time the orbital of the electron pair.
   * change the needed interaction term : Viijj => Vii(j-1)(j-1)
   *
   * The pair is now excited from orbital i to orbital j-1 instead of to orbital j.
   */
  *exc_address -= Y[ *to_incr * LminN + (*currpair)[ 0 ] ];
  --(*currpair)[ 0 ];
  --(*V);

  /* So now check if this excitation is valid.
   * Two possibilities why not valid:
   *    1) The pair I excited, ended up in an orbital that was already occupied.
   *       Since I decrease the orbital number every time and the exc_occ array is sorted from
   *       low to high, this can only happen if exc_occ[ curr ] == exc_occ[ curr - 1 ].
   *
   *    2) I excited the pair too an orbital index too low, i.e. index=-1
   *
   * If the first case is true, I do the thing in the while loop.
   * If the second case is true, i go to the if clause after this while loop, this second case can
   * only occur if *to_incr = 0
   *
   * If my excitation is OK i just return 1 and exit the function.
   */
  while( *to_incr != 0 && (*currpair)[ 0 ] == (*currpair)[ -1 ] )
  {
    /** So apparently you were in case 1:
     *  So my exc_occ looks like this : .. .. .. .. 5 5 .. .. .. for example.
     *  The succession of 5 and 5 is wrong ofcourse.
     *  The second 5 comes for example from 6 that I decreased to 5.
     *  So a pair that comes from 6 cant be placed to 5, but maybe to 4, this I fix
     *  by decreasing the *to_incr and decreasing the exc_occ[ *to_incr ]
     *  So that i get .. .. .. .. 4 5 .. .. ..
     *
     *  So first i decrease the to_incr.
     *  Then I reset the pointer currpair to the right exc_occ element.
     *  I change the exc_address appropriately.
     *  I decrease the orbital of the electron pair.
     * change the needed interaction term : Viijj => Vii(j-1)(j-1)
     *
     * and after that recheck if this was a valid move. And make the right conclusion in my while
     * loop.
     */
    --(*to_incr);
    *currpair = &exc_occ[ *to_incr ];
    *exc_address -= Y[ *to_incr * LminN + (*currpair)[ 0 ] ];
    --(*currpair)[ 0 ];
    --(*V);
  }

  if( (*currpair)[ 0 ] < 0 )
  {
    /* So apparently i was in case 2:
     * This means that the pair that I excited from orbital *augm in the original SD cant be 
     * excited further.
     * So instead i try to excite the previous orbital and see if this works.
     */
    int i;
    /* If i am already exciting the first orbital and that failed, I cant do anything more
     * and i found all the excitations from this particular SD. */
    if( *augm == 0 )
      return 0;

    /* If it was not zero, I just reinitialise exc_occ again to occ. */
    for( i = *augm ; i >= 0 ; --i ) exc_occ[ i ] = occ[ i ];
    /* I decrease augm, and say this is also the pair im going to excite, meaning im going to
     * excite the previous pair in the SD */
    *to_incr = --( *augm );
    *currpair = &exc_occ[ *to_incr ];

    /* Put the pointer of the interaction to Viiii with i = occ[ *augm ] */
    *V = &Viikk[ occ[ *augm ] * L + occ[ *augm ] ];
    /* Put the exc_address to the original SD */
    *exc_address = address;

    /* do next_bitstring but this time, i am incrementing the doci pair excited from the new 
     * orbital */
    return prev_bitstring( address, exc_address, occ, exc_occ, to_incr, augm, V, L, N, LminN, Y, 
        Viikk, currpair );
  }
  return 1;
}

static void increment_occ( int occ[], int exc_occ[], const unsigned int N, const unsigned int LminN )
{
  /* This function increments the current occupation passed to the next occupation in the array.
   * exc_occ is coppied to be exactly the same as occ.
   */
  unsigned int to_incr = N - 1;

  /* This is a bit cryptic...
   *
   * So first i start to increment the last doci pair.  
   * so to_incr is the label of the doci pair. occ[ to_incr ] is the orbital it is in.
   * So first I check in which orbital this last pair is, and check if I can still excite
   * this pair to a next orbital. This is only possible if occ[ to_incr ] - to_incr < LminN.
   * If this excitement is possible, it happens (hence the occ[ to_incr ]++ in the next line ).
   *
   * If this increment is not possible, i look at the previous doci pair if I can increment this.
   * Each doci pair with number to_incr can not be in a higher orbital than (L - N + to_incr)
   * That is the condition of the while loop and is because I need to be able to fill the following 
   * DOCI pairs in the next orbitals.
   */
  while( occ[ to_incr ] - to_incr == LminN ) --to_incr;
  ++occ[ to_incr ];

  /** All the DOCI particles that come after the particle that is excited (i.e. to_incr)
   *  are put right after the excited particle. So if particle 5 is put in orbital 9, then
   *  particle 6,7,8,9,... are put in orbital 10,11,12,13,...
   */
  for( ; to_incr < N - 1 ; ++to_incr )
    occ[ to_incr + 1 ] = occ[ to_incr ] + 1;

  /* copy of occ to exc_occ */
  for( to_incr = 0 ; to_incr < N ; ++to_incr )
    exc_occ[to_incr] = occ[to_incr];
}
