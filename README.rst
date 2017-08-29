etl-gpu-blas
############

Mini BLAS-like library for GPU (complementary to CUBLAS).

The goal of this library is principally to be used as a complement
to CUBLAS in the ETL library. The goal is to add functions that are
not present in CUBLAS and make them available in the same format.

Features
********

So far, the library supports the following features:

 * Vector sum (egblas_Xsum)
 * Vector scalar addition (egblas_scalar_Xadd)
 * Vector scalar division (egblas_scalar_Xdiv)
 * Vector element-wise sqrt (egblas_Xsqrt)
 * Vector element-wise log (egblas_Xlog)
 * y = (alpha * x) * y (egblas_Xaxmy)
 * y = (alpha * x) / y (egblas_Xaxdy)

All functions are supporting single-precision floating points (s)
and double precision floating points (d). When possible, the
functions are also supporting single precision complex floating
points (c) and double precision complex floating points (z).
