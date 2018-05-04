#ifndef DOCI_H
# define DOCI_H

/**
 * \file DOCI.h
 * \brief The DOCI header file.
 *
 * The primary file for the DOCI routine, defines the matvec product for
 * DOCI and initializes the calculations by calling the right sparse solver.
 * These are defined in file wrapper_solvers.h
 */;
double* doci_get_diagonal( void* );

void fill_doci_data( char* dumpfil, void** dat );

int doci_get_basis_size( void* );

void doci_cleanup_data( void** );

/**
 * \brief The actual matvec product for DOCI.
 *
 * This is a some of the effects of the diagonal and off-diagonal elements, the off-diagonal is
 * given by the calculate_sigma_three function.
 *
 * \param [in] vec The inputted vector for the matvec.
 * \param [out] result The resulting vector of the matvec.
 * \param [in] vdata The pointer to the matvec_data struct needed for excecution of the matvec 
 * product.
 */
void doci_matvec(double* vec, double* result, void* );

#endif
