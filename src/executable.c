#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <argp.h>

#include "lapack.h"
#include "macros.h"
#include "debug.h"

#include "doci.h"
#include "options.h"
#include "wrapper_solvers.h"

int do_doci;

/* ============================================================================================== */
/* ================================ DECLARATION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

static void initialize_program(int argc, char *argv[], char* solver, int *max_its, double *tol, 
    int *david_keep, int *david_max_vec, int *HF_init);

static void read_inputfile(char filename[], char dumpfil[], char solver[], int *max_its, double* tol, 
    int *david_keep, int *david_max_vec, int *HF_init);

static void print_inputfile(char *dumpfil, char *solver, int max_its, double tol, int david_keep, 
    int david_max_vec, int HF_init);

static double* make_init_guess(int HF_init);

static int get_basis_size(void);

static double* get_diagonal(void);

static void cleanup_data(void);

/* ============================================================================================== */

int main(int argc, char* argv[] ){
  int max_its;
  int david_keep;
  int david_max_vec;
  int HF_init;
  int basis_size;
  int info;
  double tol;
  double energy;
  double* result;
  double* diagonal;
  char solver[20];
  long long t_elapsed;
  double d_elapsed;
  struct timeval t_start, t_end;
  gettimeofday(&t_start, NULL);

  /* line by line write-out */
  setvbuf(stdout, NULL, _IOLBF, BUFSIZ);

  initialize_program(argc, argv, solver, &max_its, &tol, &david_keep, &david_max_vec, &HF_init);
  basis_size = get_basis_size();
  result = make_init_guess(HF_init);
  diagonal = get_diagonal();

  printf("--------------------------------------------------\n");
  info = sparse_eigensolve(result, basis_size, &energy, doci_matvec, diagonal, NULL, tol,
      max_its, solver, david_keep, david_max_vec);

  gettimeofday(&t_end, NULL);
  t_elapsed = (t_end.tv_sec - t_start.tv_sec) * 1000000LL + t_end.tv_usec - t_start.tv_usec;
  d_elapsed = t_elapsed*1e-6;

  printf("\t\t ** EXIT CODE SOLVER : %d\n"
         "\t\t ** %s ENERGY : %.10lf\n"
         "\t\t ** ELAPSED TIME : %f\n"
         "--------------------------------------------------\n"
         "exiting...\n",
         info, do_doci ? "DOCI" : "FCI", energy, d_elapsed);

  cleanup_data();
  safe_free(result);

  return EXIT_SUCCESS;
}

/* ============================================================================================== */
/* ================================= DEFINITION STATIC FUNCTIONS ================================ */
/* ============================================================================================== */

const char *argp_program_version = "doci 0.1";
const char *argp_program_bug_address = "<Klaas.Gunst@UGent.be>";

/* A description of the program */
static char doc[] = 
  "doci -- An implementation of DOCI (Doubly Occupied Configuration Interaction)"
  "\n\n"
  "input-file\n"
  "----------\n"
  "\n"
  "FCIDUMP          = path to the FCIDUMP file.\n"
  "\n"
  "SOLVER           = The sparse solver to be used. (D for Davidson, CG for\n"
  "                   conjugate gradient and CGP for conjugate gradient with\n"
  "                   diagonal preconditioner). (default: D)\n"
  "\n"
  "MAX_ITS          = Maximum number of iterations. (default: 200)\n"
  "\n"
  "TOL              = The tolerance for convergence. (default: 1e-8)\n"
  "\n"
  "DAVIDSON_KEEP    = The vectors to be kept after deflation in the Davidson\n"
  "                   algorithm. (default: 2)\n"
  "\n"
  "DAVIDSON_MAX_VEC = The maximum dimension of the subspace optimization before\n"
  "                   Davidson deflates. (default: 30)\n"
  "\n"
  "HF_INIT          = Set to 1 if initial guess should be the Hartree-Fock\n"
  "                   solution. Set to 0 if the initial guess should be random.\n"
  "                   (default: 1)\n";


/* A description of the arguments we accept. */
static char args_doc[] = "INPUTFILE";

/* The options we understand. */
static struct argp_option options[] = {
  { "mode", 'm', "WHICH_MODE", 0, "For Full Configuration Interaction calculations set mode to 1. "
                                  "For Double Occupied Configuration Ineraction set mode to 0. "
                                  "(default: 0)\n"},
  { 0 } /* options struct needs to be closed by a { 0 } option */
};

/* Used by main to communicate with parse_opt. */
struct arguments{
  char *inputfile;
};

/* Parse a single option. */
  static error_t parse_opt(int key, char *arg, struct argp_state *state){
  /* Get the input argument from argp_parse, which we
   *      know is a pointer to our arguments structure. */
  struct arguments *arguments = state->input;

  switch (key)
  {
    case 'm':
      do_doci = atoi(arg) ? 0 : 1;
      break;
    case ARGP_KEY_ARG:
      if (state->arg_num > 1)
        /* Too many arguments. */
        argp_usage (state);

      arguments->inputfile = arg;
      break;

    case ARGP_KEY_END:
      if (state->arg_num < 1)
        /* Not enough arguments. */
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

static void initialize_program(int argc, char *argv[], char* solver, int *max_its, double *tol, 
    int *david_keep, int *david_max_vec, int *HF_init){
  char dumpfil[255];
  struct arguments arguments;

  /* Default values. */
  arguments.inputfile = NULL;
  do_doci = 1;

  /* Parse our arguments; every option seen by parse_opt will be reflected in arguments. */
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if(!do_doci){
    fprintf(stderr, "Only DOCI works at this moment!\n");
    exit(EXIT_FAILURE);
  }

  read_inputfile(arguments.inputfile, dumpfil, solver, max_its, tol, david_keep, david_max_vec,
      HF_init);
  print_inputfile(dumpfil, solver, *max_its, *tol, *david_keep, *david_max_vec, *HF_init);

  if(do_doci)
    fill_doci_data(dumpfil);
}

static void read_inputfile(char filename[], char dumpfil[], char solver[], int *max_its, double* tol, 
    int *david_keep, int *david_max_vec, int *HF_init){
  char buffer[255];
  FILE *fp;

  fp = fopen(filename, "r");
  if(fp == NULL){
    fprintf(stderr, "Error reading input file\n");
    exit(EXIT_FAILURE);
  }

  /* default settings */
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
}

static void print_inputfile(char *dumpfil, char *solver, int max_its, double tol, int david_keep, 
    int david_max_vec, int HF_init){
  char buffer[255];

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

  printf(
      "doci -- An implementation of DOCI (Doubly Occupied Configuration Interaction)"
      "\n\n"
      "input-file\n"
      "----------\n"
      "\n"
      "FCIDUMP          = %s\n"
      "\n"
      "SOLVER           = %s\n"
      "\n"
      "MAX_ITS          = %d\n"
      "\n"
      "TOL              = %e\n"
      "\n"
      "DAVIDSON_KEEP    = %d\n"
      "\n"
      "DAVIDSON_MAX_VEC = %d\n"
      "\n"
      "INITIAL          = %s\n\n",
      dumpfil, buffer, max_its, tol, david_keep, david_max_vec, HF_init ? "Hartree-Fock" : "random"
  );
}

static double* make_init_guess(int HF_init){
  int basis_size = get_basis_size();
  double *result = safe_calloc(basis_size, double);
  int ONE = 1;
  
  if(HF_init)
    result[0] = 1;
  else{
    int cnt;
    double norm;
    srand(time(NULL));
    for(cnt = 0 ; cnt < basis_size; cnt++ ) result[cnt] = rand();
    norm = 1/dnrm2_(&basis_size, result, &ONE);
    dscal_(&basis_size, &norm, result, &ONE);
  }
  return result;
}

static int get_basis_size(void){
  if(do_doci)
    return doci_get_basis_size();
  else
    return 0;
  /*  return fci_get_basis_size(); */
}

static double* get_diagonal(void){
  if(do_doci)
    return doci_get_diagonal();
  else
    return NULL;
  /*  return fci_get_diagonal(); */
}

static void cleanup_data(void){
  if(do_doci)
    doci_cleanup_data();
}
