// /*
//  * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//  *
//  * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
//  * and proprietary rights in and to this software, related documentation
//  * and any modifications thereto.  Any use, reproduction, disclosure or
//  * distribution of this software and related documentation without an express
//  * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
//  */
//
// #include <ATen/ATen.h>
// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdexcept>
//
// using namespace Eigen;
//
// #define BLOCKS_FOR_1D_BUFFER(buffer_len, num_threads) \
//     buffer_len + (num_threads - 1) / num_threads;
//
// namespace kaolin {
//
// /** @brief A kernel for evaluating 3D gaussians.
//  *  @param X Coordinates of shape (K, 3)
//  *  @param M Gaussian means of shape (N, 3)
//  *  @param Q Gaussian cov - quaternions of shape (N, 4)
//  *  @param S Gaussian cov - scales (N, 3)
//  *  @param F Gaussian field values (N, D)
//  *  @param out_values Output: Tensor of gaussian values
//  *  @param NUM_QUERIES Total number of coordinate queries (== K)
//  *  @param NUM_GAUSSIANS Total number of gaussians (== N)
//  *  @param NUM_DIMS Total number of field dimensions (== D)
//  */
// __global__ void eval_3d_gaussian_field_cuda_kernel(
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,    // Coordinates                 (K, 3)
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> M,    // Gaussian means              (N, 3)
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Q,    // Gaussian cov - quaternions  (N, 4)
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> S,    // Gaussian cov - scales       (N, 3)
//     const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> F,    // Gaussian field values       (N, D)
//     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_values, // Tensor of gaussian values
//     const int32_t NUM_QUERIES,
//     const int32_t NUM_GAUSSIANS,
//     const int32_t NUM_DIMS
// ) {
//     const uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
//     if (tidx > NUM_GAUSSIANS)
//         return;
//
//     const auto mean = M[tidx];
//     const auto cov_q = Q[tidx];
//     const auto cov_s = S[tidx];
//     const auto field_val = F[tidx];
//
//     // Convert unit quaternion to rotation matrix
//     Quaternionf q(cov_q[0], cov_q[1], cov_q[2], cov_q[3]);
//     Matrix3f rotation_mat = q.toRotationMatrix();
//
//     Vector3f scale(cov_s[0], cov_s[1], cov_s[2]);
//     auto D = scale.array().inverse().square().matrix().asDiagonal();
//     Matrix3f inv_cov = rotation_mat * D * rotation_mat.transpose();
//
//     // TODO (@operel): optimization -- unroll const steps if queries > fixed value
//     // for (auto qidx = 0; qidx < NUM_QUERIES; qidx++) {
//     //     const auto x = X[qidx];
//     //     Vector3f x_local(x[0] - mean[0], x[1] - mean[1], x[2] - mean[2]);
//     //     const float d = x_local.transpose() * inv_cov * x_local;
//     //     const float prob = _EXP(-0.5 * d);
//     //     for (auto dim_idx = 0; dim_idx < NUM_DIMS; dim_idx++) {
//     //         atomicAdd(&(out_values[qidx][dim_idx]), field_val[dim_idx] * prob);
//     //     }
//     // }
// }
//
// void eval_gaussian_field_cuda_impl(
//     at::Tensor X,          // Coordinates           (K, 3)
//     at::Tensor M,          // Gaussian means        (N, 3)
//     at::Tensor Q,          // Gaussian quaternions  (N, 4)
//     at::Tensor S,          // Gaussian scales       (N, 3)
//     at::Tensor F,          // Gaussian field values (N, D)
//     at::Tensor out_values  // Tensor of gaussian values
// ) {
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(M));
//     TORCH_CHECK(X.ndimension() == 2);
//     TORCH_CHECK(M.ndimension() == 2);
//     TORCH_CHECK(Q.ndimension() == 2);
//     TORCH_CHECK(S.ndimension() == 2);
//     TORCH_CHECK(F.ndimension() == 2);
//     const int32_t NUM_QUERIES = X.size(0);
//     const int32_t NUM_GAUSSIANS = M.size(0);
//     const int32_t NUM_DIMS = F.size(1);
//     auto num_threads = 1024;
//     auto num_blocks = BLOCKS_FOR_1D_BUFFER(NUM_GAUSSIANS, num_threads);
//
//     eval_3d_gaussian_field_cuda_kernel<<<num_blocks, num_threads>>>(
//         X.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         M.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         Q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         S.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         F.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         out_values.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//         NUM_QUERIES,
//         NUM_GAUSSIANS,
//         NUM_DIMS
//     );
// }
//
// } // namespace kaolin
