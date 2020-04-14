// Copyright (c) 2017 Hiroharu Kato
// Copyright (c) 2018 Nikos Kolotouros
// Copyright (c) 2019 Shichen Liu

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


namespace{

template <typename scalar_t>
__device__ __forceinline__ void barycentric_coordinate(scalar_t *w, const scalar_t x, const scalar_t y, const scalar_t *face_info) {
    w[0] = face_info[3 * 0 + 0] * x + face_info[3 * 0 + 1] * y + face_info[3 * 0 + 2];
    w[1] = face_info[3 * 1 + 0] * x + face_info[3 * 1 + 1] * y + face_info[3 * 1 + 2];
    w[2] = face_info[3 * 2 + 0] * x + face_info[3 * 2 + 1] * y + face_info[3 * 2 + 2];
}


template <typename scalar_t>
__device__ __forceinline__ bool check_border(const scalar_t x, const scalar_t y, const scalar_t *face, const scalar_t threshold) {
    return (x > max(max(face[0], face[3]), face[6]) + threshold ||
            x < min(min(face[0], face[3]), face[6]) - threshold ||
            y > max(max(face[1], face[4]), face[7]) + threshold ||
            y < min(min(face[1], face[4]), face[7]) - threshold);
}


template <typename scalar_t>
__device__ __forceinline__ bool check_face_frontside(const scalar_t *face) {
    return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
}


template <typename scalar_t>
__device__ __forceinline__ bool check_pixel_inside(const scalar_t *w) {
    return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
}


template <typename scalar_t>
__device__ __forceinline__ void barycentric_clip(scalar_t *w) {
    for (int k = 0; k < 3; k++) w[k] = max(min(w[k], 1.), 0.);
    const scalar_t w_sum = max(w[0] + w[1] + w[2], 1e-5);
    for (int k = 0; k < 3; k++) w[k] /= w_sum;
}


template <typename scalar_t>
__device__ __forceinline__ void euclidean_p2f_distance(scalar_t &sign, scalar_t &dis_x, scalar_t &dis_y,
                                                       scalar_t *w, scalar_t *t, 
                                                       const scalar_t* face, const scalar_t *face_info,
                                                       const scalar_t xp, const scalar_t yp) {
    const scalar_t *face_sym = face_info + 9;
    const scalar_t *face_obt = face_info + 18;

    if (w[0] > 0 && w[1] > 0 && w[2] > 0 &&
        w[0] < 1 && w[1] < 1 && w[2] < 1) {
        // inside the triangle, w[0] + w[1] + w[2] = 0
        scalar_t dis_min = 100000000;
        scalar_t dis_x_min = 0;
        scalar_t dis_y_min = 0;
        scalar_t a0[3];
        scalar_t t0[3];
        for (int k = 0; k < 3; k++) {
            int v0 = k;
            int v1 = (k + 1) % 3;
            int v2 = (k + 2) % 3;
            a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
            a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
            a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

            t0[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
            t0[v1] = 1 - t0[v0];
            t0[v2] = 0;

            t0[0] -= w[0];
            t0[1] -= w[1];
            t0[2] -= w[2];

            // calculate distance
            dis_x = t0[0] * face[0] + t0[1] * face[3] + t0[2] * face[6];
            dis_y = t0[0] * face[1] + t0[1] * face[4] + t0[2] * face[7];
            scalar_t dis = dis_x * dis_x + dis_y * dis_y;

            if (dis < dis_min) {
                dis_min = dis;
                dis_x_min = dis_x;
                dis_y_min = dis_y;
                t[0] = t0[0];
                t[1] = t0[1];
                t[2] = t0[2];
            }
        }
        dis_x = dis_x_min;
        dis_y = dis_y_min;
        sign = 1;
    } else {
        int v0 = -1;

        if (w[1] <= 0 && w[2] <= 0) {
            v0 = 0;
            if (face_obt[0] == 1 && (xp - face[0]) * (face[6] - face[0]) + (yp - face[1]) * (face[7] - face[1]) > 0) v0 = 2;
        } else if (w[2] <= 0 && w[0] <= 0) {
            v0 = 1;
            if (face_obt[1] == 1 && (xp - face[3]) * (face[0] - face[3]) + (yp - face[4]) * (face[1] - face[4]) > 0) v0 = 0;
        } else if (w[0] <= 0 && w[1] <= 0) {
            v0 = 2;
            if (face_obt[2] == 1 && (xp - face[6]) * (face[3] - face[6]) + (yp - face[7]) * (face[4] - face[7]) > 0) v0 = 1;
        } else
        if (w[0] <= 0) v0 = 1;
        else if (w[1] <= 0) v0 = 2;
        else if (w[2] <= 0) v0 = 0;

        const int v1 = (v0 + 1) % 3;
        const int v2 = (v0 + 2) % 3;

        scalar_t a0[3];

        a0[0] = face_sym[3 * v0 + 0] - face_sym[3 * v1 + 0];
        a0[1] = face_sym[3 * v0 + 1] - face_sym[3 * v1 + 1];
        a0[2] = face_sym[3 * v0 + 2] - face_sym[3 * v1 + 2];

        t[v0] = (w[0] * a0[0] + w[1] * a0[1] + w[2] * a0[2] - a0[v1]) / (a0[v0] - a0[v1]);
        t[v1] = 1 - t[v0];
        t[v2] = 0;

        // clamp to [0, 1]
        for (int k = 0; k < 3; k++) {
            t[k] = min(max(t[k], 0.), 1.);
            t[k] -= w[k];
        }

        // calculate distance
        dis_x = t[0] * face[0] + t[1] * face[3] + t[2] * face[6];
        dis_y = t[0] * face[1] + t[1] * face[4] + t[2] * face[7];
        sign = -1;
    }
}


template <typename scalar_t>
__device__ __forceinline__ void forward_barycentric_p2f_distance(scalar_t &dis, const scalar_t *w) {
    dis = w[0] > w[1] ? (w[1] > w[2] ? w[2] : w[1]) : (w[0] > w[2] ? w[2] : w[0]);
    dis = dis > 0 ? pow(dis, 2) : -pow(dis, 2);
}


template <typename scalar_t>
__device__ __forceinline__ void backward_barycentric_p2f_distance(scalar_t grad_v[3][3], const scalar_t *w, const scalar_t *face_info, const scalar_t xp, const scalar_t yp, const scalar_t dis, const scalar_t C) {
    const int p = w[0] > w[1] ? (w[1] > w[2] ? 2 : 1) : (w[0] > w[2] ? 2 : 0);
    const scalar_t *face_inv = face_info;
    for (int l = 0; l < 2; l++) {
        for (int k = 0; k < 3; k++) {
            scalar_t grad_kl = 0;
            for (int q = 0; q < 3; q++) {
                grad_kl += -face_inv[3*p+l] * face_inv[3*k+q] * (q == 0 ? xp : (q == 1 ? yp : 1));
            }
            grad_v[k][l] = grad_kl * C;
            grad_v[k][l] *= dis > 0 ? (2. * sqrt(dis)) : (2. * sqrt(-dis));
        }
    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t forward_sample_texture(const scalar_t *texture, const scalar_t *w, const int R, const int k, const int texture_sample_type) {
    scalar_t texture_k;
    if (texture_sample_type == 0) { // sample surface color with resolution as R
        const int w_x = w[0] * R;
        const int w_y = w[1] * R;
        if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
            texture_k = texture[(w_y * R + w_x) * 3 + k];
        } else {
            texture_k = texture[((R - 1 - w_y) * R + (R - 1 - w_x)) * 3 + k];
        }
    } else
    if (texture_sample_type == 1) { // sample vertex color
        texture_k = w[0] * texture[k] + w[1] * texture[3+k] + w[2] * texture[6+k];
    }
    return texture_k;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t backward_sample_texture(const scalar_t grad_color, const scalar_t *w, const int R, const int k, const int texture_sample_type) {
    scalar_t grad_texture_k;
    if (texture_sample_type == 0) { // sample surface color with resolution as R
        const int w_x = w[0] * R;
        const int w_y = w[1] * R;
        if ((w[0] + w[1]) * R - w_x - w_y <= 1) {
            if (k == w_y * R + w_x) {
                grad_texture_k = grad_color;
            }
        } else {
            if (k == (R - 1 - w_y) * R + (R - 1 - w_x)) {
                grad_texture_k = grad_color;
            }
        }
    } else
    if (texture_sample_type == 1) {
        grad_texture_k = w[k] * grad_color;
    }
    return grad_texture_k;
}


// triangle preprocessing
template <typename scalar_t>
__global__ void forward_soft_rasterize_inv_cuda_kernel(
        const scalar_t* __restrict__ faces,
        scalar_t* faces_info,
        int batch_size,
        int num_faces,
        int image_size) {
    /* batch number, face, number, image size, face[v012][RGB] */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * num_faces) {
        return;
    }
    // const int is = image_size;
    const scalar_t* face = &faces[i * 9];
    scalar_t* face_inv = &faces_info[i * 27];
    scalar_t* face_sym = &faces_info[i * 27+9];
    scalar_t* face_obt = &faces_info[i * 27+18];

    /* return if backside */
    // if ((face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]))
        // return;
    /* p[num][xy]: x, y is (-1, 1). */
    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        for (int dim = 0; dim < 2; dim++) {
            p[num][dim] = face[3 * num + dim]; // no normalize
        }
    }
    /* compute face_inv */
    scalar_t face_inv_star[9] = {
        p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
        p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
        p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_determinant = (
        p[2][0] * (p[0][1] - p[1][1]) +
        p[0][0] * (p[1][1] - p[2][1]) +
        p[1][0] * (p[2][1] - p[0][1]));
    face_inv_determinant = face_inv_determinant > 0 ? max(face_inv_determinant, 1e-10) : min(face_inv_determinant, -1e-10);
    /* set to global memory */
    for (int k = 0; k < 9; k++) {
        face_inv[k] = face_inv_star[k] / face_inv_determinant;
    }
    /* F * F.T */
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            face_sym[j * 3 + k] = face[j * 3 + 0] * face[k * 3 + 0] +
                                  face[j * 3 + 1] * face[k * 3 + 1] + 
                                  1;
        }
    }
    /* check if one arc is obt arc */
    for (int k = 0; k < 3; k++) {
        const int k0 = k;
        const int k1 = (k + 1) % 3;
        const int k2 = (k + 2) % 3;
        if ((p[k1][0] - p[k0][0]) * (p[k2][0] - p[k0][0]) + (p[k1][1] - p[k0][1]) * (p[k2][1] - p[k0][1]) < 0) {
            face_obt[k0] = 1;
            break;
        }
    }
}


template <typename scalar_t>
__global__ void forward_soft_rasterize_cuda_kernel(
        const scalar_t* __restrict__ faces,
        const scalar_t* __restrict__ textures,
        const scalar_t* __restrict__ faces_info,
        scalar_t* aggrs_info,
        scalar_t* soft_colors,
        int batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        int texture_res,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    ////////////////////////
    ////////////////////////

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = is - 1 - (pn / is);
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1. - is) / is;
    const scalar_t xp = (2. * xi + 1. - is) / is;

    const scalar_t *face = &faces[bn * nf * 9] - 9;
    const scalar_t *texture = &textures[bn * nf * texture_size * 3] - texture_size * 3;
    const scalar_t *face_info = &faces_info[bn * nf * 27] - 27;

    const scalar_t threshold = dist_eps * sigma_val;

    // Initialize pixel color
    scalar_t soft_color[4] = {1., 1., 1., 0.};
    if (func_id_alpha == 2) soft_color[3] = 1.;
    scalar_t softmax_sum = exp(eps / gamma_val);
    scalar_t softmax_max = eps;
    for (int k = 0; k < 3; k++) {
        if (func_id_rgb == 0) { // hard assign, set to background
            soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn];
        } else
        if (func_id_rgb == 1) {
            soft_color[k] = soft_colors[(bn * 4 + k) * (is * is) + pn] * softmax_sum; // initialize background color
        }
    }
    scalar_t depth_min = 10000000;
    int face_index_min = -1;

    for (int fn = 0; fn < nf; fn++) {
        face += 9;
        texture += texture_size * 3;
        face_info += 27;

        if (check_border(xp, yp, face, sqrt(threshold))) continue; // triangle too far away from pixel

        scalar_t dis;
        scalar_t dis_x;
        scalar_t dis_y;
        scalar_t t[3];
        scalar_t w[3];
        scalar_t w_clip[3];
        scalar_t sign;
        scalar_t soft_fragment;

        // compute barycentric coordinate w
        barycentric_coordinate(w, xp, yp, face_info);

        // compute probability map based on distance functions
        if (func_id_dist == 0) { // hard assign
            soft_fragment = check_pixel_inside(w) ? 1. : 0.;
            if (soft_fragment == 0.) continue; // ignore triangle outside of the pixel
        } else
        if (func_id_dist == 1) { // barycentric distance
            forward_barycentric_p2f_distance(dis, w);
            if (-dis >= threshold) continue; // ignore triangle far away from the pixel
            soft_fragment = 1. / (1. + exp(-dis / sigma_val));
        } else
        if (func_id_dist == 2) { // euclidean distance
            euclidean_p2f_distance(sign, dis_x, dis_y, w, t, face, face_info, xp, yp);
            dis = dis_x * dis_x + dis_y * dis_y;
            if (sign < 0 && dis >= threshold) continue; // ignore triangle far away from the pixel
            soft_fragment = 1. / (1. + exp(-sign * dis / sigma_val));
        }

        /////////////////////////////////////////////////////

        // aggragate for alpha channel
        if (func_id_alpha == 0) { // hard assign
            if (soft_fragment > 0.5) soft_color[3] = 1.;
        } else
        if (func_id_alpha == 1) { // Sum
            soft_color[3] += soft_fragment;
        } else 
        if (func_id_alpha == 2) { // Logical-Or
            soft_color[3] *= 1. - soft_fragment;
        }

        /////////////////////////////////////////////////////

        for (int k = 0; k < 3; k++) w_clip[k] = w[k];
        barycentric_clip(w_clip);
        const scalar_t zp = 1. / (w_clip[0] / face[2] + w_clip[1] / face[5] + w_clip[2] / face[8]);
        if (zp < near || zp > far) continue; // triangle out of screen, pass

        /////////////////////////////////////////////////////
        // aggregate for rgb channels
        if (func_id_rgb == 0) { // Hard assign
            if (zp < depth_min && check_pixel_inside(w) && (double_side || check_face_frontside(face))) {
                depth_min = zp;
                face_index_min = fn;
                for (int k = 0; k < 3; k++) {
                    soft_color[k] = forward_sample_texture(texture, w_clip, texture_res, k, texture_sample_type);
                }
            }
        } else
        if (func_id_rgb == 1) { // D * Softmax (Z)
            if (check_face_frontside(face) || double_side) {
                const scalar_t zp_norm =  (far - zp) / (far - near);
                scalar_t exp_delta_zp = 1.;
                if (zp_norm > softmax_max) {
                    exp_delta_zp = exp((softmax_max - zp_norm) / gamma_val);
                    softmax_max = zp_norm;
                }
                const scalar_t exp_z = exp((zp_norm - softmax_max) / gamma_val);
                softmax_sum = exp_delta_zp * softmax_sum + exp_z * soft_fragment;
                for (int k = 0; k < 3; k++) {
                    const scalar_t color_k = forward_sample_texture(texture, w_clip, texture_res, k, texture_sample_type);
                    soft_color[k] = exp_delta_zp * soft_color[k] + exp_z * soft_fragment * color_k;// * soft_fragment;
                }
            }
        }
    }

    //////////////////////////////////////////////

    // finalize aggregation
    if (func_id_alpha == 0) {
        soft_colors[(bn * 4 + 3) * (is * is) + pn] =  soft_color[3];
    } else
    if (func_id_alpha == 1) {
        soft_colors[(bn * 4 + 3) * (is * is) + pn] =  soft_color[3] / nf;
    } else 
    if (func_id_alpha == 2) {
        soft_colors[(bn * 4 + 3) * (is * is) + pn] =  1. - soft_color[3];
    }

    if (func_id_rgb == 0) {
        if (face_index_min != -1)
            for (int k = 0; k < 3; k++) {
                soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k];
            }
        aggrs_info[(bn * 2 + 0) * (is * is) + pn] = depth_min;
        aggrs_info[(bn * 2 + 1) * (is * is) + pn] = face_index_min;
    } else
    if (func_id_rgb == 1) {
        for (int k = 0; k < 3; k++) {
            soft_colors[(bn * 4 + k) * (is * is) + pn] = soft_color[k] / softmax_sum;
        }
        aggrs_info[(bn * 2 + 0) * (is * is) + pn] = softmax_sum;
        aggrs_info[(bn * 2 + 1) * (is * is) + pn] = softmax_max;
    }
}


template <typename scalar_t>
__global__ void backward_soft_rasterize_cuda_kernel(
        const scalar_t* __restrict__ faces,
        const scalar_t* __restrict__ textures,
        const scalar_t* __restrict__ soft_colors,
        const scalar_t* __restrict__ faces_info,
        const scalar_t* __restrict__ aggrs_info, // 0: sum, 1: max z*D
        scalar_t* grad_faces,
        scalar_t* grad_textures,
        scalar_t* grad_soft_colors,
        int batch_size,
        int num_faces,
        int image_size,
        int texture_size,
        int texture_res,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    ////////////////////////
    ////////////////////////

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size * image_size) {
        return;
    }
    const int is = image_size;
    const int nf = num_faces;
    const int bn = i / (is * is);
    const int pn = i % (is * is);
    const int yi = is - 1 - (pn / is);
    const int xi = pn % is;
    const scalar_t yp = (2. * yi + 1 - is) / is;
    const scalar_t xp = (2. * xi + 1 - is) / is;

    const scalar_t* face = &faces[bn * nf * 9] - 9;
    const scalar_t* texture = &textures[bn * nf * texture_size * 3] - texture_size * 3;
    const scalar_t* face_info = &faces_info[bn * nf * 27] - 27;

    const scalar_t threshold = dist_eps * sigma_val;

    const scalar_t softmax_sum = aggrs_info[(bn * 2 + 0) * (is * is) + pn];
    const scalar_t softmax_max = aggrs_info[(bn * 2 + 1) * (is * is) + pn];

    for (int fn = 0; fn < nf; fn++) {
        face += 9;
        texture += texture_size * 3;
        face_info += 27;

        if (check_border(xp, yp, face, sqrt(threshold))) continue;

        scalar_t dis;
        scalar_t dis_x;
        scalar_t dis_y;
        scalar_t t[3];
        scalar_t w[3];
        scalar_t sign;
        scalar_t soft_fragment;

        barycentric_coordinate(w, xp, yp, face_info);

        // compute probability map based on distance functions
        if (func_id_dist == 0) { // hard assign
            soft_fragment = check_pixel_inside(w) ? 1. : 0.;
            if (soft_fragment == 0.) continue; // ???
        } else
        if (func_id_dist == 1) { // barycentric distance
            forward_barycentric_p2f_distance(dis, w);
            for (int k = 0; k < 3; k++) t[k] = w[k];
            if (-dis >= threshold) continue; // ignore triangle far away from the pixel
            soft_fragment = 1. / (1. + exp(-dis / sigma_val));
        } else
        if (func_id_dist == 2) { // euclidean distance
            euclidean_p2f_distance(sign, dis_x, dis_y, w, t, face, face_info, xp, yp);
            dis = dis_x * dis_x + dis_y * dis_y;
            if (sign < 0 && dis >= threshold) continue; // ignore triangle too far away from the pixel, sigmoid(-9) = 0.000123
            soft_fragment = 1. / (1. + exp(-sign * dis / sigma_val));
        }


        scalar_t* grad_face = &grad_faces[(bn * nf + fn) * 9];
        scalar_t* grad_texture = &grad_textures[(bn * nf + fn) * texture_size];
        scalar_t grad_v[3][3] = {0};
        scalar_t C_grad_xy = 0;

        /////////////////////////////////////////////////////

        // aggragate for alpha channel
        scalar_t C_grad_xy_alpha = grad_soft_colors[(bn * 4 + 3) * (is * is) + pn];
        if (func_id_alpha == 0) { // hard assign
            // hard assign alpha channels does not have gradient
        } else
        if (func_id_alpha == 1) { // Sum
            C_grad_xy_alpha /= nf;
        } else 
        if (func_id_alpha == 2) { // Logical-Or
            C_grad_xy_alpha *= (1 - soft_colors[(bn * 4 + 3) * (is * is) + pn]) / max(1 - soft_fragment, 1e-6);
        }
        C_grad_xy += C_grad_xy_alpha;

        /////////////////////////////////////////////////////
        barycentric_clip(w);
        const scalar_t zp = 1. / (w[0] / face[2] + w[1] / face[5] + w[2] / face[8]);
        if (zp < near || zp > far) continue; // triangle out of screen, pass

        // aggregate for rgb channels
        if (func_id_rgb == 0) { // Hard assign, no gradient to xyz
            if (fn == softmax_max) {
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < texture_size; j++) {
                        atomicAdd(&grad_texture[k], backward_sample_texture(grad_soft_colors[(bn * 4 + k) * (is * is) + pn], w, texture_res, j, texture_sample_type));
                    }
                }
            }
        } else
        if (func_id_rgb == 1 && (check_face_frontside(face) || double_side)) { // Softmax (Z * D)
            scalar_t C_grad_xyz_rgb = 0.;

            const scalar_t zp_norm = (far - zp) / (far - near);
            const scalar_t zp_softmax = soft_fragment * exp((zp_norm - softmax_max) / gamma_val) / softmax_sum;

            for (int k = 0; k < 3; k++) {
                const scalar_t grad_soft_color_k = grad_soft_colors[(bn * 4 + k) * (is * is) + pn];

                for (int j = 0; j < texture_size; j++) {
                    const scalar_t grad_t = backward_sample_texture(grad_soft_color_k, w, texture_res, j, texture_sample_type);
                    atomicAdd(&grad_texture[k], zp_softmax * grad_t);
                }

                const scalar_t color_k = forward_sample_texture(texture, w, texture_res, k, texture_sample_type);
                C_grad_xyz_rgb += grad_soft_color_k * (color_k - soft_colors[(bn * 4 + k) * (is * is) + pn]);// * soft_fragment;
            }
            C_grad_xyz_rgb *= zp_softmax;
            C_grad_xy += C_grad_xyz_rgb / soft_fragment;

            const scalar_t C_grad_z_rgb = C_grad_xyz_rgb / gamma_val / (near - far) * zp * zp;
            grad_v[0][2] = C_grad_z_rgb * w[0] / face[2] / face[2];
            grad_v[1][2] = C_grad_z_rgb * w[1] / face[5] / face[5];
            grad_v[2][2] = C_grad_z_rgb * w[2] / face[8] / face[8];
        }

        /////////////////////////////////////////////////////

        C_grad_xy *= soft_fragment * (1 - soft_fragment) / sigma_val; // sigmoid gradient
        // compute probability map gradient based on distance functions
        if (func_id_dist == 1) { // barycentric distance
            backward_barycentric_p2f_distance(grad_v, t, face_info, xp, yp, dis, C_grad_xy);
        } else
        if (func_id_dist == 2) { // euclidean distance
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 2; l++) {
                    grad_v[k][l] = 2 * sign * C_grad_xy * (t[k] + w[k]) * (l == 0 ? dis_x : dis_y);
                }
            }
        }

        atomicAdd(&grad_face[0], grad_v[0][0]);// * sigma_val);
        atomicAdd(&grad_face[1], grad_v[0][1]);// * sigma_val);
        atomicAdd(&grad_face[3], grad_v[1][0]);// * sigma_val);
        atomicAdd(&grad_face[4], grad_v[1][1]);// * sigma_val);
        atomicAdd(&grad_face[6], grad_v[2][0]);// * sigma_val);
        atomicAdd(&grad_face[7], grad_v[2][1]);// * sigma_val);

        atomicAdd(&grad_face[2], grad_v[0][2]);// * gamma_val);
        atomicAdd(&grad_face[5], grad_v[1][2]);// * gamma_val);
        atomicAdd(&grad_face[8], grad_v[2][2]);// * gamma_val);
    }
}


}


std::vector<at::Tensor> forward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const auto texture_res = int(sqrt(texture_size));
    const int threads = 512;
    const dim3 blocks_1 ((batch_size * num_faces - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_soft_rasterize_inv_cuda", ([&] {
      forward_soft_rasterize_inv_cuda_kernel<scalar_t><<<blocks_1, threads>>>(
          faces.data<scalar_t>(),
          faces_info.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in forward_transform_inv_triangle: %s\n", cudaGetErrorString(err));

    const dim3 blocks_2 ((batch_size * image_size * image_size - 1) / threads +1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "forward_eff_soft_rasterize_cuda", ([&] {
      forward_soft_rasterize_cuda_kernel<scalar_t><<<blocks_2, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          faces_info.data<scalar_t>(),
          aggrs_info.data<scalar_t>(),
          soft_colors.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size,
          texture_res,
          near,
          far,
          eps,
          sigma_val,
          func_id_dist,
          dist_eps,
          gamma_val,
          func_id_rgb,
          func_id_alpha,
          texture_sample_type,
          double_side);
      }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_soft_rasterize: %s\n", cudaGetErrorString(err));

    return {faces_info, aggrs_info, soft_colors};
}


std::vector<at::Tensor> backward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,        
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    const auto batch_size = faces.size(0);
    const auto num_faces = faces.size(1);
    const auto texture_size = textures.size(2);
    const auto texture_res = int(sqrt(texture_size));
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size * image_size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "backward_soft_rasterize_cuda", ([&] {
      backward_soft_rasterize_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          textures.data<scalar_t>(),
          soft_colors.data<scalar_t>(),
          faces_info.data<scalar_t>(),
          aggrs_info.data<scalar_t>(),
          grad_faces.data<scalar_t>(),
          grad_textures.data<scalar_t>(),
          grad_soft_colors.data<scalar_t>(),
          batch_size,
          num_faces,
          image_size,
          texture_size,
          texture_res,
          near,
          far,
          eps,
          sigma_val,
          func_id_dist,
          dist_eps,
          gamma_val,
          func_id_rgb,
          func_id_alpha,
          texture_sample_type,
          double_side);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in backward_soft_rasterize: %s\n", cudaGetErrorString(err));

    return {grad_faces, grad_textures};
}