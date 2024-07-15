///*
// * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
// *
// * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// * and proprietary rights in and to this software, related documentation
// * and any modifications thereto.  Any use, reproduction, disclosure or
// * distribution of this software and related documentation without an express
// * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
// */
//
//#include <ATen/ATen.h>
//
//namespace kaolin {
//
//void eval_gaussian_field_cuda_impl(
//    at::Tensor X,         // Coordinates           (K, 3)
//    at::Tensor M,         // Gaussian means        (N, 3)
//    at::Tensor Q,         // Gaussian quaternions  (N, 4)
//    at::Tensor S,         // Gaussian scales       (N, 3)
//    at::Tensor F,         // Gaussian field values (N, D)
//    at::Tensor out_values // Output: Tensor of gaussian values at coordinates
//);
//
///* =========================================== */
//
//at::Tensor eval_gaussian_field(
//    at::Tensor X,   // Coordinates           (K, 3)
//    at::Tensor M,   // Gaussian means        (N, 3)
//    at::Tensor Q,   // Gaussian quaternions  (N, 4)
//    at::Tensor S,   // Gaussian scales       (N, 3)
//    at::Tensor F    // Gaussian field values (N, D)
//) {
//#ifdef WITH_CUDA
//  int64_t num_queries = X.size(0);
//  int64_t num_dims = F.size(1);
//  at::Tensor queries_out = at::zeros({num_queries, num_dims}, M.options().dtype(at::kFloat));
//  eval_gaussian_field_cuda_impl(X, M, Q, S, F, queries_out);
//  return queries_out;
//#else
//  AT_ERROR(__func__);
//#endif  // WITH_CUDA
//}
//
//} // namespace kaolin
