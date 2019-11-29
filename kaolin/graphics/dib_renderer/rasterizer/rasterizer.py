# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function
from __future__ import division

import torch
import torch.nn
import torch.autograd
from torch.autograd import Function

import kaolin.graphics.dib_renderer.cuda.rasterizer as cuda_rasterizer

import cv2
import numpy as np
import datetime


@torch.jit.script
def prepare_tfpoints(tfpoints3d_bxfx9, tfpoints2d_bxfx6,
                     multiplier: float, batch_size: int, num_faces: int, expand: float):
    # avoid numeric error
    tfpoints2dmul_bxfx6 = multiplier * tfpoints2d_bxfx6

    # bbox
    tfpoints2d_bxfx3x2 = tfpoints2dmul_bxfx6.view(batch_size, num_faces, 3, 2)
    tfpoints_min = torch.min(tfpoints2d_bxfx3x2, dim=2)[0]
    tfpoints_max = torch.max(tfpoints2d_bxfx3x2, dim=2)[0]
    tfpointsbbox_bxfx4 = torch.cat((tfpoints_min, tfpoints_max), dim=2)

    # bbox2
    tfpoints_min = tfpoints_min - expand * multiplier
    tfpoints_max = tfpoints_max + expand * multiplier
    tfpointsbbox2_bxfx4 = torch.cat((tfpoints_min, tfpoints_max), dim=2)

    # depth
    _tfpoints3d_bxfx9 = tfpoints3d_bxfx9.permute(2, 0, 1)
    tfpointsdep_bxfx1 = (_tfpoints3d_bxfx9[2, :, :] +
                         _tfpoints3d_bxfx9[5, :, :] +
                         _tfpoints3d_bxfx9[8, :, :]).unsqueeze(-1) / 3.0

    return tfpoints2dmul_bxfx6, tfpointsbbox_bxfx4, tfpointsbbox2_bxfx4, tfpointsdep_bxfx1


class LinearRasterizer(Function):
    @staticmethod
    def forward(ctx,
                width,
                height,
                tfpoints3d_bxfx9,
                tfpoints2d_bxfx6,
                tfnormalz_bxfx1,
                vertex_attr_bxfx3d,
                expand=None,
                knum=None,
                multiplier=None,
                delta=None,
                debug=False):

        if expand is None:
            expand = 0.02
        if knum is None:
            knum = 30
        if multiplier is None:
            multiplier = 1000
        if delta is None:
            delta = 7000

        batch_size = tfpoints3d_bxfx9.shape[0]
        num_faces = tfpoints3d_bxfx9.shape[1]

        num_vertex_attr = vertex_attr_bxfx3d.shape[2] / 3
        assert num_vertex_attr == int(
            num_vertex_attr), \
            'vertex_attr_bxfx3d has shape {} which is not a multiple of 3' \
            .format(vertex_attr_bxfx3d.shape[2])

        num_vertex_attr = int(num_vertex_attr)

        ###################################################
        start = datetime.datetime.now()

        tfpoints2dmul_bxfx6, tfpointsbbox_bxfx4, tfpointsbbox2_bxfx4, tfpointsdep_bxfx1 = \
            prepare_tfpoints(tfpoints3d_bxfx9, tfpoints2d_bxfx6,
                             multiplier, batch_size, num_faces, expand)

        device = tfpoints2dmul_bxfx6.device

        # output
        tfimidxs_bxhxwx1 = torch.zeros(
            batch_size, height, width, 1, dtype=torch.float32, device=device)
        # set depth as very far
        tfimdeps_bxhxwx1 = torch.full(
            (batch_size, height, width, 1), fill_value=-1000., dtype=torch.float32, device=device)
        tfimweis_bxhxwx3 = torch.zeros(
            batch_size, height, width, 3, dtype=torch.float32, device=device)
        tfims_bxhxwxd = torch.zeros(
            batch_size, height, width, num_vertex_attr, dtype=torch.float32, device=device)
        tfimprob_bxhxwx1 = torch.zeros(
            batch_size, height, width, 1, dtype=torch.float32, device=device)

        # intermidiate varibales
        tfprobface = torch.zeros(
            batch_size, height, width, knum, dtype=torch.float32, device=device)
        tfprobcase = torch.zeros(
            batch_size, height, width, knum, dtype=torch.float32, device=device)
        tfprobdis = torch.zeros(batch_size, height, width,
                                knum, dtype=torch.float32, device=device)
        tfprobdep = torch.zeros(batch_size, height, width,
                                knum, dtype=torch.float32, device=device)
        tfprobacc = torch.zeros(batch_size, height, width,
                                knum, dtype=torch.float32, device=device)

        # face direction
        tfpointsdirect_bxfx1 = tfnormalz_bxfx1.contiguous()
        cuda_rasterizer.forward(tfpoints3d_bxfx9,
                                tfpoints2dmul_bxfx6,
                                tfpointsdirect_bxfx1,
                                tfpointsbbox_bxfx4,
                                tfpointsbbox2_bxfx4,
                                tfpointsdep_bxfx1,
                                vertex_attr_bxfx3d,
                                tfimidxs_bxhxwx1,
                                tfimdeps_bxhxwx1,
                                tfimweis_bxhxwx3,
                                tfprobface,
                                tfprobcase,
                                tfprobdis,
                                tfprobdep,
                                tfprobacc,
                                tfims_bxhxwxd,
                                tfimprob_bxhxwx1,
                                multiplier,
                                delta)

        end = datetime.datetime.now()
        ###################################################

        if debug:
            print(end - start)
            ims_bxhxwxd = tfims_bxhxwxd.detach().cpu().numpy()
            improbs_bxhxwx1 = tfimprob_bxhxwx1.detach().cpu().numpy()
            imidxs_bxhxwx1 = tfimidxs_bxhxwx1.detach().cpu().numpy()
            imdeps_bxhxwx1 = tfimdeps_bxhxwx1.detach().cpu().numpy()
            imweis_bxhxwx3 = tfimweis_bxhxwx3.detach().cpu().numpy()

            print(ims_bxhxwxd.shape)
            print(improbs_bxhxwx1.shape)
            print(np.max(improbs_bxhxwx1))

            cv2.imshow("0", ims_bxhxwxd[-1, :, :, :3])
            cv2.imshow("1", improbs_bxhxwx1[-1])
            cv2.imshow("2", imweis_bxhxwx3[-1])
            cv2.imshow("3", imidxs_bxhxwx1[-1] / num_faces)
            cv2.imshow("4", imdeps_bxhxwx1[-1])
            cv2.waitKey()

        debug_im = torch.zeros(batch_size, height, width, 3,
                               dtype=torch.float32, device=device)

        ctx.save_for_backward(tfims_bxhxwxd, tfimprob_bxhxwx1,
                              tfimidxs_bxhxwx1, tfimweis_bxhxwx3,
                              tfpoints2dmul_bxfx6, vertex_attr_bxfx3d,
                              tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc,
                              debug_im)

        ctx.multiplier = multiplier
        ctx.delta = delta
        ctx.debug = debug

        tfims_bxhxwxd.requires_grad = True
        tfimprob_bxhxwx1.requires_grad = True

        return tfims_bxhxwxd, tfimprob_bxhxwx1

    @staticmethod
    def backward(ctx, dldI_bxhxwxd, dldp_bxhxwx1):
        tfims_bxhxwxd, tfimprob_bxhxwx1, \
            tfimidxs_bxhxwx1, tfimweis_bxhxwx3, \
            tfpoints2dmul_bxfx6, tfcolors_bxfx3d, \
            tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc, \
            debug_im = ctx.saved_variables

        multiplier = ctx.multiplier
        delta = ctx.delta
        debug = ctx.debug
        # avoid numeric error
        # multiplier = 1000
        # tfpoints2d_bxfx6 *= multiplier

        dldp2 = torch.zeros_like(tfpoints2dmul_bxfx6)
        dldp2_prob = torch.zeros_like(tfpoints2dmul_bxfx6)
        dldc = torch.zeros_like(tfcolors_bxfx3d)
        cuda_rasterizer.backward(dldI_bxhxwxd.contiguous(),
                                 dldp_bxhxwx1.contiguous(),
                                 tfims_bxhxwxd, tfimprob_bxhxwx1,
                                 tfimidxs_bxhxwx1, tfimweis_bxhxwx3,
                                 tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc,
                                 tfpoints2dmul_bxfx6, tfcolors_bxfx3d,
                                 dldp2, dldc, dldp2_prob,
                                 debug_im, multiplier, delta)
        if debug:
            print(dldc[dldc > 0.1])
            print(dldc[dldc > 0.1].shape)
            print(dldp2[dldp2 > 0.1])
            print(dldp2[dldp2 > 0.1].shape)
            print(dldp2_prob[dldp2_prob > 0.1])
            print(dldp2_prob[dldp2_prob > 0.1].shape)

        return \
            None, \
            None, \
            None, \
            dldp2 + dldp2_prob, \
            None, \
            dldc, \
            None, \
            None, \
            None, \
            None, \
            None, \
            None


linear_rasterizer = LinearRasterizer.apply
