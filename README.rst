etl-gpu-blas (egblas)
#####################

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
 * Vector element-wise exp (egblas_Xexp)
 * y = (alpha * x) * y (egblas_Xaxmy)
 * y = (alpha * x) / y (egblas_Xaxdy)

All functions are supporting single-precision floating points (s)
and double precision floating points (d). When possible, the
functions are also supporting single precision complex floating
points (c) and double precision complex floating points (z).

Synchronization
***************

By default, most of the kernels executed by this library are not
synchronized. In the future, no kernel will be synchronized. If you
want to synchronize after the function call, you can use
`cudaDeviceSynchronize()` after the egblas function call. If you
want all egblas functions to be synchronized, you can define
EGBLAS_SYNCHRONIZE::

    EXTRA_CXX_FLAGS=-DEGBLAS_SYNCHRONIZE make

In that case, every egblas function will be terminated by
a `cudaDeviceSynchronize()` call. This can have a big performance
impact, especially if working on small collections of data, since
the kernel launch has a high overhead.
