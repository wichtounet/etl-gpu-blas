etl-gpu-blas
############

Mini BLAS-like library for GPU (complementary to CUBLAS).

The goal of this library is principally to be used as a complement
to CUBLAS in the ETL library. The goal is to add functions that are
not present in CUBLAS and make them available in the same format.

Features
********

So far, the library has very few features:
 * Vector sum (egblas_Xsum)
 * Vector scalar addition (egblas_scalar_Xadd)

Both functions are supported in the single-precision and
double-precision modes.
