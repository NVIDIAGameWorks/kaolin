/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef KAOLIN_OPS_CONVERSIONS_UNBATCHED_MCUBE_DEFINES_H_
#define KAOLIN_OPS_CONVERSIONS_UNBATCHED_MCUBE_DEFINES_H_

namespace kaolin {

typedef unsigned int uint;
typedef unsigned char uchar;

}  // namespace kaolin

// if SAMPLE_VOLUME is 0, an implicit dataset is generated. If 1, a voxelized
// dataset is loaded from file
#define SAMPLE_VOLUME 1

// Using shared to store computed vertices and normals during triangle generation
// improves performance
#define USE_SHARED 1

// The number of threads to use for triangle generation (limited by shared memory size)
#define NTHREADS 32

#define SKIP_EMPTY_VOXELS 1

#endif  // KAOLIN_OPS_CONVERSIONS_UNBATCHED_MCUBE_DEFINES_H_
