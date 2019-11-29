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

import torch
import torch.nn
import torch.autograd
from torch.autograd import Function

import numpy as np
import cv2

######################################################
if __name__ == '__main__':
    
    from graphics.utils.utils_mesh import loadobj, face2pfmtx, loadobjtex
    from graphics.utils.utils_perspective import lookatnp, perspectiveprojectionnp
    from graphics.utils.utils_sphericalcoord import get_spherical_coords_x
    
    from graphics.render.texrender import TexRender
    
    meshfile = './data/banana.obj'
    p, f, uv, ft = loadobjtex(meshfile)
    pfmtx = face2pfmtx(f)
    
    '''
    p, f = loadobj('2.obj')
    uv = get_spherical_coords_x(p)
    uv[:, 0] = -uv[:, 0]
    ft = f
    '''
    
    # make it fit pytorch coordinate
    # uv[:, 1] = 1 - uv[:, 1]
    # uv = uv * 2 - 1
    
    imfile = './data/banana.jpg'
    texturenp = cv2.imread(imfile)[:, :, ::-1].astype(np.float32) / 255.0
    
    ##################################################################
    pmax = np.max(p, axis=0, keepdims=True)
    pmin = np.min(p, axis=0, keepdims=True)
    pmiddle = (pmax + pmin) / 2
    p = p - pmiddle
    
    coef = 5
    p = p * coef
    
    ##########################################################
    campos = np.array([0, 0, 1.5], dtype=np.float32)  # where camera it is
    camcenter = np.array([0, 0, 0], dtype=np.float32)  # where camra is looking at
    camup = np.array([-1, 1, 0], dtype=np.float32)  # y axis of camera view
    camviewmtx, camviewshift = lookatnp(campos.reshape(3, 1), camcenter.reshape(3, 1), camup.reshape(3, 1))
    camviewshift = -np.dot(camviewmtx.transpose(), camviewshift)
    
    camfovy = 45 / 180.0 * np.pi
    camprojmtx = perspectiveprojectionnp(camfovy, 1.0 * 1.0 / 1.0)
    
    #####################################################
    tfp_px3 = torch.from_numpy(p)
    tfp_px3.requires_grad = True
    
    tff_fx3 = torch.from_numpy(f)
    
    tfuv_tx2 = torch.from_numpy(uv)
    tfuv_tx2.requires_grad = True
    tfft_fx3 = torch.from_numpy(ft)
    
    tftex_thxtwx3 = torch.from_numpy(np.ascontiguousarray(texturenp))
    tftex_thxtwx3.requires_grad = True
    
    tfcamviewmtx = torch.from_numpy(camviewmtx)
    tfcamshift = torch.from_numpy(camviewshift)
    tfcamproj = torch.from_numpy(camprojmtx)
    
    ##########################################################
    tfp_1xpx3 = torch.unsqueeze(tfp_px3, dim=0)
    tfuv_1xtx2 = torch.unsqueeze(tfuv_tx2, dim=0)
    tftex_1xthxtwx3 = torch.unsqueeze(tftex_thxtwx3, dim=0)
    
    tfcamviewmtx_1x3x3 = torch.unsqueeze(tfcamviewmtx, dim=0)
    tfcamshift_1x3 = tfcamshift.view(-1, 3)
    tfcamproj_3x1 = tfcamproj
    
    bs = 4
    tfp_bxpx3 = tfp_1xpx3.repeat([bs, 1, 1])
    tfuv_bxtx2 = tfuv_1xtx2.repeat([bs, 1, 1])
    tftex_bxthxtwx3 = tftex_1xthxtwx3.repeat([bs, 1, 1, 1])
    
    tfcamviewmtx_bx3x3 = tfcamviewmtx_1x3x3.repeat([bs, 1, 1])
    tfcamshift_bx3 = tfcamshift_1x3.repeat([bs, 1])
    tfcameras = [tfcamviewmtx_bx3x3.cuda(), \
                 tfcamshift_bx3.cuda(), \
                 tfcamproj_3x1.cuda()]
    
    # tfcameras = None
    tftex_bx3xthxtw = tftex_bxthxtwx3.permute([0, 3, 1, 2])
    renderer = TexRender(256, 256)
    tfim_bxhxwx3, _, _ = renderer.forward(points=[tfp_bxpx3.cuda(), tff_fx3.cuda()], \
                                          cameras=tfcameras, \
                                          colors=[tfuv_bxtx2.cuda(), tfft_fx3.cuda(), tftex_bx3xthxtw.cuda()])
    
    loss1 = torch.sum(tfim_bxhxwx3)
    print('loss im {}', format(loss1.item()))
    
    im_hxwx3 = tfim_bxhxwx3.detach().cpu().numpy()[-1]
    cv2.imshow("", im_hxwx3[:, :, ::-1])
    cv2.waitKey()
    
    loss1.backward()
    
    print(tfp_px3.grad[tfp_px3.grad > 0])
    
    np.save(file='gt.npy', arr=im_hxwx3)

