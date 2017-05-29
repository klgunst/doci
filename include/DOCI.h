#ifndef DOCI_H
# define DOCI_H

/**
 * \file DOCI.h
 * \brief The DOCI header file.
 *
 * The primary file for the DOCI routine, defines the matvec product for
 * DOCI and initializes the calculations by calling the right sparse solver.
 * These are defined in file wrapper_solvers.h
 */


/* DEBUGGING FUNCTIONS */
/**
 * \brief Debugging function, prints Y (array for bitstring-to-cnt conversion).
 * \param [in] Y The array for bitstring-to-cnt conversion.
 * \param [in] N Number of electron pairs.
 * \param [in] NminL width of Y array (L - N + 1).
 */
void print_Y(unsigned int* Y, const int N, const int NminL);
/**
 * \brief Debugging function, prints different bitstrings and their cnt.
 * \param [in] bitstrings The list of bitstrings.
 * \param [in] size The number of bitstrings.
 * \param [in] L The number of orbitals.
 */
void print_bitstrings(unsigned int* bitstrings, const unsigned int size, const int L);
/**
 * \brief Debugging function, prints a bitstring.
 * \param bitstring The bitstring to be printed.
 * \param L Number of orbitals.
 */
void print_bitstring(unsigned int bitstring, const int L);
/**
 * \brief Debugging function, checks the cnt-to-bitstring conversion.
 * \param [in] bitstrings The list of bitstrings to be checked.
 * \param [in] Y Array for bitstring-to-cnt conversion.
 * \param [in] N Number of electron pairs.
 * \param [in] L Number of orbitals.
 */
void check_bitstrings(unsigned int* bitstrings, unsigned int* Y, const int N, const int L);
/**
 * \brief Debugging function, checks the next_bitstring function.
 * \param [in] to_bitstrings List of bitstrings.
 * \param [in] Y Array for bitstring-to-cnt conversion.
 * \param [in] N Number of electron pairs.
 * \param [in] L Number of orbitals.
 */
void check_next_bitstring(unsigned int* to_bitstrings, unsigned int* Y, int N, int L);
/* END OF DEBUGGING FUNCTIONS */

unsigned int* initialize_Y1(const int, const int);

unsigned int* initialize_x(const int, const int);

unsigned int combination(const int, const int);

unsigned int* initialize_to_bitstring(const int, const int);

/**
 * \brief This function is responsible the conversion of a basis state to its cnt number.
 *
 * This function does essentially the same as occ_to_cnt but is bit slower because the occupations
 * still need to be calculated. The other one is used in the computational intensive part 
 * of the routine.
 * 
 * \param [in] bitstring The bitstring that needs to be converted.
 * \param [in] Y Array that helps conversion of bitstring to cnt number of a basis state.
 * \param [in] N Number of electron pairs.
 * \param [in] L Number of orbitals.
 * \returns The cnt number of the converted bitstring.
 */
unsigned int bitstring_to_cnt(unsigned int bitstring, unsigned int* Y, const int N, const int L);

/**
 * \brief Calculates the double excited terms for the matvec products. Thus both alpha and beta are
 *        excited to the same orbital.
 * 
 * \param [in] vec The inputted state of the matvec.
 * \param [in] two_p_int The two particle integrals.
 * \param [in] to_bitstring A list for conversion of cnt number to the corresponding bitstring.
 * \param [in] Y Array that helps conversion of bitstring to cnt number of a basis state.
 * \param [in] N Number of electron pairs.
 * \param [in] L Number of orbitals.
 * \param [out] result The resulting state of the matvec.
 */
void calculate_sigma_three(double* vec, double* two_p_int, unsigned int* to_bitstring,\
    unsigned int* Y, int, const int N, const int L, double* result);

/**
 * \brief This function gives the next excited bitstring from a beginning bitstring and the sign 
 *        corresponding with the excititation (in DOCI this sign is always 1).
 *
 * \param [in] cnt The cnt number of the state to be excited.
 * \param [in,out] exc_cnt The previous excited cnt number is inputted here and the next one is
 *                         outputted here.
 * \param [in,out] i the previous orbital excited to is input, the next orbital excited to is output.
 * \param [in,out] j the previous orbital excited from is input, the next orbital excited from is 
 *                   output.
 * \param [in] occ Array with the occupation numbers of the orginal state.
 * \param [in,out] exc_occ Array with the occupation numbers of the previous and new excited state.
 * \param [in] Y Array that helps conversion of bitstring to cnt number of a basis state.
 * \param [in] L Number of orbitals.
 * \param [in] N Number of electron pairs.
 * \return Returns the sign of the excitation (for DOCI always +1) or a zero when no new excitations
 *         can been given.
 */
int next_bitstring(int cnt, int* exc_cnt, int* i, int* j, int* occ, int* exc_occ, unsigned int* Y, \
    const int L, const int N);

/**
 * \brief This function is responsible the conversion of a occupation array to its cnt number.
 *
 * \param [in] occ The occupations of the orbitals representing the bitstring that needs to be
 *                 converted.
 * \param [in] Y Array that helps conversion of bitstring to cnt number of a basis state.
 * \param [in] N Number of electron pairs.
 * \param [in] L Number of orbitals.
 * \returns The cnt number of the converted bitstring.
 */
int occ_to_cnt(int* occ, unsigned int* Y, const int N, const int L);

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
void matvec(double* vec, double* result, void* vdata);

/**
 * \brief Transforms the read integrals from a fullci dump properly to chemical notation integrals
 *        and some preprocessing for the single-particle integrals.
 */
void transform_integrals(double*, double*, int);

/**
 * \brief creates the diagonal elements of the Hamiltonian.
 */
double* create_diagonal(unsigned int*, int, const int, double*, double*, double);

/**
 * \brief Fills an array of occupation numbers for the given bitstring.
 * 
 * \param [in] bitstring The bitstring of which occupation numbers for the orbitals are asked for.
 * \param [out] occ The array where storage of the occupation numbers should happen, array should 
 *                  already be allocated.
 * \param [in] ORBS Number of orbitals.
 */
void fill_occupation(unsigned int bitstring, int* occ, const int ORBS);
#endif
