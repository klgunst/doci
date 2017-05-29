#ifndef DAVIDSON_H
# define DAVIDSON_H

/**
 * \file davidson.h
 * \brief The Davidson header file.
 *
 * This file contains the Davidson optimization with diagonal preconditioner.
 */


/**
 * \brief main function for Davidson algorithm.
 * \param [in] data Pointer to a data structure needed for the matvec function.
 * \param [in,out] result Initial guess as input, converged vector as output.
 * \param [out] energy The converged energy.
 * \param [in] max_vectors Maximum number of vectors kept before deflation happens.
 * \param [in] keep_deflate Number of vectors kept after deflation.
 * \param [in] davidson_tol The tolerance.
 * \param [in] matvec The pointer to the matrix vector product.
 * \param [in] diagonal Diagonal elements of the Hamiltonian.
 * \param [in] basis_size The dimension of the problem.
 * \param [in] max_its Maximum number of iterations.
 */
int davidson(void* data, double* result, double* energy, int max_vectors, int keep_deflate, \
    double davidson_tol, void (*matvec)(double*, double*, void*), double* diagonal, int basis_size,\
    int max_its);

void new_search_vector(double*, double*, int, int);

void expand_submatrix(double*, double*, double*, int, int, int);

double calculate_residue(double*, double*, double*, double, double*, double*, int, int);

void create_new_vec_t(double*, double*, double, int);

void deflate(double*, double*, double*, int, int, int, double*, double*, void (*matvec)(double*, \
      double*, void*), void*);

#endif
