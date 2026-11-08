#ifndef APPLICATION_PILOT_DEFAULT_H
#define APPLICATION_PILOT_DEFAULT_H
#ifdef __cplusplus
extern "C" {
#endif
/* CSR is zero-based. x is an input initial guess and output solution.
 * stats: import seconds, setup seconds, solve seconds, reported relative
 * residual, iterations, native solve error. cg=1 is a reference solve only. */
int pilot_default(int n, const int *ia, const int *ja, const double *a,
                  const double *b, double *x, double tol, int maxit,
                  int cg, double *stats);
#ifdef __cplusplus
}
#endif
#endif
