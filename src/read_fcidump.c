#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "debug.h"
#include "read_fcidump.h"
#include "options.h"
void read_inputfile(char filename[], char dumpfil[], char solver[], int *max_its, double* tol, \
                  int *david_keep, int *david_max_vec, int *HF_init){
  char buffer[255];
  FILE *fp;

  fp = fopen(filename, "r");
  if(fp==NULL){
    fprintf(stderr, "Error reading input file\n");
    exit(EXIT_FAILURE);
  }

  *max_its = SOLVER_MAX_ITS;
  strcpy(solver, "D");
  *tol = SOLVER_TOL;
  *david_keep = DAVIDSON_KEEP_DEFLATE;
  *david_max_vec = DAVIDSON_MAX_VECS;
  *HF_init = 1;

  while(fgets(buffer, sizeof buffer, fp) != NULL){
    sscanf(buffer, " FCIDUMP = %s ", dumpfil);
    sscanf(buffer, " SOLVER = %s ", solver);
    sscanf(buffer, " MAX_ITS = %d ", max_its);
    sscanf(buffer, " TOL = %lf ", tol);
    sscanf(buffer, " DAVIDSON_KEEP = %d ", david_keep);
    sscanf(buffer, " DAVIDSON_MAX_VEC = %d ", david_max_vec);
    sscanf(buffer, " HF_INIT = %d ", HF_init);
  }
  fclose(fp);

  if(strcmp(solver, "D") == 0)
    strcpy(buffer, "DAVIDSON");
  else if(strcmp(solver, "CG") == 0)
    strcpy(buffer, "CONJUGATE GRADIENT");
  else if(strcmp(solver, "CGP") == 0) 
    strcpy(buffer, "CONJUGATE GRADIENT WITH DIAGONAL PRECONDITIONER");
  else{
    printf("no correct solver inputted!\n");
    exit(EXIT_FAILURE);
  }

  /* PRINTING OPTIONS INPUTTED */
  printf("***************************\n");
  printf("********* OPTIONS *********\n");
  printf("***************************\n\n\n");

  printf("FCIDUMP : %s\n\n", dumpfil);
  printf("SOLVER : %s\n\n", buffer);
  printf("MAX_ITS : %d\n\n", *max_its);
  printf("TOL : %e\n\n", *tol);
  if(strcmp_ign_ws(solver, "D") == 0){
    printf("DAVIDSON_KEEP_DEFLATE : %d\n\n", *david_keep);
    printf("DAVIDSON_KEEP_MAX_VEC : %d\n\n", *david_max_vec);
  }
  printf("HF_INIT : %d\n\n", *HF_init);
}

void read_fcidump(char filename[], int *NORB, int *NALPHA, int * NBETA, double* core_energy, \
                  double** one_p_int, double** two_p_int){
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
  fgets(buffer, sizeof buffer, fp);
  cnt = sscanf(buffer, " &FCI NORB= %d , NELEC= %d , MS2= %d , ", NORB, &N, &MS2);
  
  if(cnt != 3){
    fprintf(stderr, "Error in reading %s. File is wrongly formatted.\n", filename);
    exit(EXIT_FAILURE);
  }

  *NALPHA = ( N + MS2 ) / 2;
  *NBETA = ( N - MS2 ) / 2;

  printf("ORBITALS : %d\n\n", *NORB);
  printf("ELECTRONS : %d\n\n", N);
  printf("MS2 : %d\n\n", MS2);
  printf("***************************\n");

  *core_energy = 0;
  
  *one_p_int = (double *) calloc( sizeof(double), (*NORB) * (*NORB));
  *two_p_int = (double *) calloc(sizeof(double), (*NORB) * (*NORB) * (*NORB) * (*NORB));

  /* at this moment im skipping point symmetry */
  while(fgets(buffer, sizeof buffer, fp) != NULL){
    ln_cnt++;
    if(strcmp_ign_ws(buffer, "&END") == 0 || strcmp_ign_ws(buffer, "/END") == 0 || \
        strcmp_ign_ws(buffer, "/") == 0)
      break;
  }

  while(fgets(buffer, sizeof buffer, fp) != NULL){
    cnt = sscanf(buffer, " %lf %d %d %d %d ", &value, &i, &j, &k, &l); /* chemical notation */
    ln_cnt++;
    if( cnt != 5 ){
      fprintf(stderr, "Whilst reading the integrals, an error occured, wrong formatting at line "\
         "%d!\n", ln_cnt);
      exit(EXIT_FAILURE);
    }
    
    if( k != 0 )
      matrix_el = *two_p_int + (l-1) * (*NORB) * (*NORB) * (*NORB) + (k-1) * (*NORB) * (*NORB) \
                  + (j-1) * (*NORB) +(i-1);
    else if (i != 0)
      matrix_el = *one_p_int + (j-1) * (*NORB) + (i-1);
    else 
      matrix_el = core_energy;

    if(*matrix_el != 0) fprintf(stderr, "Doubly inputted value at line %d, hope you don\'t mind\n",\
        ln_cnt);
    *matrix_el = value;
  }
  
  fclose(fp);
}

int strcmp_ign_ws(const char *s1, const char *s2){
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
