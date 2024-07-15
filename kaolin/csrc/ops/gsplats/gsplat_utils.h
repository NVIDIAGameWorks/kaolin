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
//#pragma once
//
//#include <ATen/ATen.h>
//
//namespace kaolin {
//
//at::Tensor eval_gaussian_field(
//    at::Tensor X,   // Coordinates           (K, 3)
//    at::Tensor M,   // Gaussian means        (N, 3)
//    at::Tensor Q,   // Gaussian quaternions  (N, 4)
//    at::Tensor S,   // Gaussian scales       (N, 3)
//    at::Tensor F    // Gaussian field values (N, D)
//);
//
//} // namespace kaolin
