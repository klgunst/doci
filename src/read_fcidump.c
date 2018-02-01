#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "debug.h"
#include "read_fcidump.h"
#include "macros.h"

/* ============================================================================================== */
/* ================================ DECLARATION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

static int strcmp_ign_ws(const char *s1, const char *s2);

/* ============================================================================================== */

void read_fcidump(char filename[], unsigned int *NORB, unsigned int *NALPHA, unsigned int * NBETA,
    double* core_energy, double** one_p_int, double** two_p_int){
  char buffer[255];
  int cnt, N, MS2, i, j, k, l, ln_cnt;
  double *matrix_el;
  double value;
  FILE *fp;

  fp = fopen(filename, "r");
  if(fp==NULL){
    fprintf(stderr, "Error reading fcidump file\n");
    exit(EXIT_FAILURE);
  }

  ln_cnt = 1;
  if(fgets(buffer, sizeof buffer, fp) == NULL){
    fprintf(stderr, "Error in reading %s. File is wrongly formatted.\n", filename);
    exit(EXIT_FAILURE);
  }
  cnt = sscanf(buffer, " &FCI NORB= %u , NELEC= %d , MS2= %d , ", NORB, &N, &MS2);
  
  if(cnt != 3){
    fprintf(stderr, "Error in reading %s. File is wrongly formatted.\n", filename);
    exit(EXIT_FAILURE);
  }

  *NALPHA = ( N + MS2 ) / 2;
  *NBETA = ( N - MS2 ) / 2;

  *core_energy = 0;
  
  *one_p_int = safe_calloc((*NORB) * (*NORB), double);
  *two_p_int = safe_calloc((*NORB) * (*NORB) * (*NORB) * (*NORB), double);

  /* at this moment im skipping point symmetry */
  while(fgets(buffer, sizeof buffer, fp) != NULL){
    ln_cnt++;
    if(strcmp_ign_ws(buffer, "&END") == 0 || strcmp_ign_ws(buffer, "/END") == 0 || 
        strcmp_ign_ws(buffer, "/") == 0)
      break;
  }

  while(fgets(buffer, sizeof buffer, fp) != NULL){
    cnt = sscanf(buffer, " %lf %d %d %d %d ", &value, &i, &j, &k, &l); /* chemical notation */
    ln_cnt++;
    if( cnt != 5 ){
      fprintf(stderr, "Whilst reading the integrals, an error occured, wrong formatting at line "
         "%d!\n", ln_cnt);
      exit(EXIT_FAILURE);
    }
    
    if( k != 0 )
      matrix_el = *two_p_int + (l-1) * (*NORB) * (*NORB) * (*NORB) + (k-1) * (*NORB) * (*NORB)
                  + (j-1) * (*NORB) +(i-1);
    else if (i != 0)
      matrix_el = *one_p_int + (j-1) * (*NORB) + (i-1);
    else 
      matrix_el = core_energy;

    if(fabs(*matrix_el) >= 1e-14) fprintf(stderr, "Doubly inputted value at line %d, hope you"
        " don\'t mind\n", ln_cnt);
    *matrix_el = value;
  }
  fclose(fp);
}

/* ============================================================================================== */
/* ================================= DEFINITION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

static int strcmp_ign_ws(const char *s1, const char *s2){
  const unsigned char *p1 = (const unsigned char *)s1;
  const unsigned char *p2 = (const unsigned char *)s2;
  
  while (*p1){
    while (isspace(*p1)) p1++;
    if (!*p1) break;
                                      
    while (isspace(*p2)) p2++;
    /*if (!*p2) break;*/
    
    if (!*p2) return  1;
    if (*p2 > *p1) return -1;
    if (*p1 > *p2) return  1;

    p1++;
    p2++;
  }
  
  if (*p2) return -1;
  
  return 0;
}
