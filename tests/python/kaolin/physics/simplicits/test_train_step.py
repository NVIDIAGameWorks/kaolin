# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest

import torch 
import numpy as np 

from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.physics.simplicits.train import train_step 
from functools import partial

# SDFs
#------------------------
def sdBox(p):
    SDBOXSIZE  = [1,1,1]
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return  np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0],max(q[1],q[2])),0.0)

# TODO: Split this into its own util maybe? or add to kaolin? 
def example_unit_cube_object(num_points=100000, yms=1e5, prs=0.45, rhos=1000):
    uniform_points = np.random.uniform([-1,-1,-1], [1,1,1], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdBox, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]

    return X0, X0_sdfval, YMs, PRs, Rhos, appx_vol

# TODO: removed the "torch.double" test because 
# (1) not critical (if it works with torch.float, then torch.double is even higher precision)
# (2) getting weird pytorch errors that are solved for pytorch v2.3
# Err: RuntimeError: Tensors of the same index must be on the same device
# and the same dtype except `step` tensors that can be CPU and float32 notwithstanding
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_train_step(device, dtype):
    NUM_HANDLES = 10
    NUM_STEPS = 10
    LR_START = 1e-3
    NUM_SAMPLES = 1000

    so_pts, _, so_yms, so_prs, so_rhos, so_appx_vol = example_unit_cube_object()
    so_model = SimplicitsMLP(3, 64, NUM_HANDLES, 6)
    so_optimizer = torch.optim.Adam(so_model.parameters(), LR_START)
    so_bb_pts = (np.min(so_pts, axis=0), np.max(so_pts, axis=0))
    so_normalized_pts = (so_pts - so_bb_pts[0])/(so_bb_pts[1] - so_bb_pts[0])

    so_pts = torch.tensor(so_pts, device=device, dtype=dtype)
    so_yms = torch.tensor(so_yms, device=device, dtype=dtype).unsqueeze(-1)
    so_prs = torch.tensor(so_prs, device=device, dtype=dtype).unsqueeze(-1)
    so_rhos = torch.tensor(so_rhos, device=device, dtype=dtype).unsqueeze(-1)
    so_normalized_pts = torch.tensor(so_normalized_pts, device=device, dtype=dtype)
    so_model.to(device=device, dtype=dtype)

    #train_step(step, model, optim, normalized_pts, yms, prs, rhos, BATCH_SIZE, NUM_HANDLES, APPX_VOL, NUM_SAMPLES, LE_COEFF, LO_COEFF)
    partial_train_step = partial(train_step, 
                                batch_size=10, 
                                num_handles=10, 
                                appx_vol=so_appx_vol, 
                                num_samples=NUM_SAMPLES, 
                                le_coeff=1e-1, 
                                lo_coeff=1e6)


    so_model.train()
    list_of_en = []
    for i in range(NUM_STEPS):
        so_optimizer.zero_grad()
        le, lo = partial_train_step(so_model, so_normalized_pts, so_yms, so_prs, so_rhos, float(i/NUM_STEPS))
        loss = le + lo
        loss.backward()
        so_optimizer.step()

        list_of_en.append(loss.item())

    so_model.eval()
    assert list_of_en[0]>list_of_en[-1]