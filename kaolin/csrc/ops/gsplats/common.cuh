// #pragma once
//
// #include <pybind11/eigen.h>
// #include <Eigen/Eigenvalues>
// #include <math.h>
// #include <cub/cub.cuh>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/ATen.h>

// typedef unsigned int uint;

// #ifdef __CUDACC__
// #define _EXP(x) __expf(x) // faster exp
// #else
// #define _EXP(x) expf(x)
// #endif

// #define DEVICE_GUARD(_tensor) \
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(_tensor));

// #define BLOCKS_FOR_1D_BUFFER(buffer_len, num_threads) \
//     buffer_len + (num_threads - 1) / num_threads;
