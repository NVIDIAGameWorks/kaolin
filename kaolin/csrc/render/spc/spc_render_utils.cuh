// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once 

#ifdef WITH_CUDA

// Get component sign for the direction ray
static __inline__ __device__ float3 ray_sgn(
    const float3 dir     // ray direction
) {
    return make_float3(
            signbit(dir.x) ? 1.0f : -1.0f,
            signbit(dir.y) ? 1.0f : -1.0f,
            signbit(dir.z) ? 1.0f : -1.0f);
}

static __inline__ __device__ float3 ray_flip(
    const float3 dir
) {
    return make_float3(-dir.x, -dir.y, -dir.z);
}

 static __inline__ __device__ float3 ray_invert(
    const float3 dir
) {
    // Prevent singularities
    const float eps = 1e-8;
    return make_float3(1.0 / (dir.x+eps),
                       1.0 / (dir.y+eps),
                       1.0 / (dir.z+eps));
}

// Device primitive for a single ray-AABB intersection
 static __inline__ __device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 sgn,    // sgn bits
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    // From Majercik et. al 2018

    // Put the center of the AABB at the origin [0, 0, 0]
    float3 o = make_float3(query.x-origin.x, query.y-origin.y, query.z-origin.z);

    // Maximum Component
    float cmax = fmaxf(fmaxf(fabs(o.x), fabs(o.y)), fabs(o.z));    


    // If the maximum component is smaller than the radius of the AABB, then the ray origin is
    // inside the AABB; return a negative to indicate.
    float winding = cmax < r ? -1.0f : 1.0f;
    winding *= r;
    if (winding < 0) {
        return winding;
    }

    // Compute distance to planes
    float d0 = fmaf(winding, sgn.x, - o.x) * invdir.x;
    float d1 = fmaf(winding, sgn.y, - o.y) * invdir.y;
    float d2 = fmaf(winding, sgn.z, - o.z) * invdir.z;
    float ltxy = fmaf(dir.y, d0, o.y);
    float ltxz = fmaf(dir.z, d0, o.z);
    float ltyx = fmaf(dir.x, d1, o.x);
    float ltyz = fmaf(dir.z, d1, o.z);
    float ltzx = fmaf(dir.x, d2, o.x);
    float ltzy = fmaf(dir.y, d2, o.y);

    // Test hit against each plane
    bool test0 = (d0 >= 0.0f) && (fabs(ltxy) <= r) && (fabs(ltxz) <= r);
    bool test1 = (d1 >= 0.0f) && (fabs(ltyx) <= r) && (fabs(ltyz) <= r);
    bool test2 = (d2 >= 0.0f) && (fabs(ltzx) <= r) && (fabs(ltzy) <= r);

    float3 _sgn = make_float3(0.0f, 0.0f, 0.0f);

    if (test0) { _sgn.x = sgn.x; }
    else if (test1) { _sgn.y = sgn.y; }
    else if (test2) { _sgn.z = sgn.z; }

    float d = 0.0f;
    if (_sgn.x != 0.0f) { d = d0; } 
    else if (_sgn.y != 0.0f) { d = d1; }
    else if (_sgn.z != 0.0f) { d = d2; }
    if (d != 0.0f) {
        return d;
    }

    return 0.0; 
    // returns: 
    //      d == 0 -> miss
    //      d >  0 -> distance
    //      d <  0 -> inside
}

static __inline__ __device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 sgn = ray_sgn(dir);
    return ray_aabb(query, dir, invdir, sgn, origin, r);
}

static __inline__ __device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 sgn = ray_sgn(dir);
    float3 invdir = ray_invert(dir);
    return ray_aabb(query, dir, invdir, sgn, origin, r);
}

static __inline__ __device__ float2 ray_aabb_with_exit(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 entry_sgn = ray_sgn(dir);
    float3 exit_sgn = ray_sgn(ray_flip(dir));
    float entry = ray_aabb(query, dir, invdir, entry_sgn, origin, r);
    float exit = ray_aabb(query, dir, invdir, exit_sgn, origin, r);
    return make_float2(entry, exit);
}

static __inline__ __device__ float2 ray_aabb_with_exit(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 invdir = ray_invert(dir);
    return ray_aabb_with_exit(query, dir, invdir, origin, r);
}

#endif //WITH_CUDA

