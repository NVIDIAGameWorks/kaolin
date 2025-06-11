# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import logging
import warnings
import warp as wp
import warp.sparse as wps
import numpy as np
import torch
import kaolin

from functools import partial

from kaolin.physics.simplicits.losses import compute_losses
from kaolin.physics.simplicits.network import SimplicitsMLP

import kaolin.physics.utils.warp_utilities as warp_utilities
import kaolin.physics.utils.torch_utilities as torch_utilities

from kaolin.physics.common import Collision, Gravity, Floor, Boundary
from kaolin.physics.materials import NeohookeanElasticMaterial
from kaolin.physics.materials.material_utils import compute_defo_grad, to_lame
from kaolin.physics.common.optimization import newtons_method
from kaolin.physics.simplicits.precomputed import sparse_lbs_matrix, sparse_dFdz_matrix_from_dense
from kaolin.physics.simplicits.skinning import weight_function_lbs


logger = logging.getLogger(__name__)

__all__ = [
    'SimplicitsObject',
    'SimulatedObject',
    'SimplicitsScene',
]

class NormalizedSkinningWeightsFcn(torch.nn.Module):
    def __init__(self, model, bb_min, bb_max):
        super().__init__()
        self.model = model
        self.bb_min = bb_min
        self.bb_max = bb_max

    def forward(self, pts):
        return torch.cat([
            self.model((pts - self.bb_min) / (self.bb_max - self.bb_min)),
            torch.ones((pts.shape[0], 1), device=pts.device)
        ], dim=1)

class SkinningWeightsFcn(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pts):
        return torch.cat([
            self.model(pts),
            torch.ones((pts.shape[0], 1), device=pts.device)
        ], dim=1)

class SimplicitsObject:
    @staticmethod
    def create_trained(pts, yms, prs, rhos, appx_vol,
                       num_handles=10,
                       num_samples=1000,
                       model_layers=6,
                       training_batch_size=10,
                       training_num_steps=10000,
                       training_lr_start=1e-3,
                       training_lr_end=1e-3,
                       training_le_coeff=1e-1,
                       training_lo_coeff=1e6,
                       training_log_every=1000,
                       normalize_for_training=True):
        r"""Constructs a SimplicitsObject by training a neural network to learn skinning weights.

        This method creates a SimplicitsObject by training a neural network to learn skinning weights
        that can be used for deformation. The network is trained to minimize a combination of
        local and global energy terms.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            appx_vol (torch.Tensor): Approximate volume tensor of shape (1,)
            num_handles (int, optional): Number of control handles for deformation. Defaults to 10
            num_samples (int, optional): Number of samples used for training. Defaults to 1000
            model_layers (int, optional): Number of layers in the neural network. Defaults to 6
            training_batch_size (int, optional): Batch size for training. Defaults to 10
            training_num_steps (int, optional): Number of training iterations. Defaults to 10000
            training_lr_start (float, optional): Initial learning rate. Defaults to 1e-3
            training_lr_end (float, optional): Final learning rate. Defaults to 1e-3
            training_le_coeff (float, optional): Coefficient for local energy term. Defaults to 1e-1
            training_lo_coeff (float, optional): Coefficient for global energy term. Defaults to 1e6
            training_log_every (int, optional): Logging frequency during training. Defaults to 1000
            normalize_for_training (bool, optional): Whether to normalize points to unit cube for training. Defaults to True

        Returns:
            SimplicitsObject: A trained SimplicitsObject with learned skinning weights

        Note:
            If num_handles is set to 0, the object will be created as rigid instead of deformable.
            The training process uses a combination of local and global energy terms to ensure
            both local detail preservation and global shape maintenance.
        """
        if num_handles == 0:
            warnings.warn(
                f'Num Handles is 0. Simplicits Object will be created as rigid.', UserWarning)

            return SimplicitsObject.create_rigid(pts, yms, prs, rhos, appx_vol)
        
        if not torch.is_tensor(yms):
            yms = torch.full((pts.shape[0],), yms, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(prs):
            prs = torch.full((pts.shape[0],), prs, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(rhos):
            rhos = torch.full((pts.shape[0],), rhos, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(appx_vol):
            appx_vol = torch.tensor([appx_vol], dtype=pts.dtype, device=pts.device)

        device = pts.device

        bb_max = torch.max(pts, dim=0).values
        bb_min = torch.min(pts, dim=0).values
        bb_vol = (bb_max[0] - bb_min[0]) * (bb_max[1] -
                                            bb_min[1]) * (bb_max[2] - bb_min[2])

        # normalize the points
        if (normalize_for_training):
            # Normalize the appx vol of object
            norm_bb_max = torch.max((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_max of the normalized pts
            norm_bb_min = torch.min((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_min of the normalized pts

            norm_bb_vol = (norm_bb_max[0] - norm_bb_min[0]) * (norm_bb_max[1] -
                                                               norm_bb_min[1]) * (norm_bb_max[2] - norm_bb_min[2])
            normalized_pts = (pts - bb_min) / (bb_max - bb_min)
            norm_appx_vol = appx_vol * (norm_bb_vol / bb_vol)

            # Set pts, appx_vol, yms, prs, rhos to normalized values
            training_pts = normalized_pts
            training_appx_vol = norm_appx_vol
        else:
            training_pts = pts
            training_appx_vol = appx_vol

        training_yms = yms.unsqueeze(-1)
        training_prs = prs.unsqueeze(-1)
        training_rhos = rhos.unsqueeze(-1)

        ######### Train the model #########
        model = SimplicitsMLP(3, 64, num_handles, model_layers)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), training_lr_start)

        model.train()
        e_logs = []
        for i in range(training_num_steps):
            optimizer.zero_grad()
            le, lo = compute_losses(model, 
                                    training_pts, 
                                    training_yms, 
                                    training_prs,
                                    training_rhos, 
                                    float(i / training_num_steps), 
                                    le_coeff=training_le_coeff, 
                                    lo_coeff=training_lo_coeff,
                                    batch_size=training_batch_size, 
                                    num_handles=num_handles,
                                    appx_vol=training_appx_vol,
                                    num_samples=num_samples)
            loss = le + lo

            loss.backward()
            optimizer.step()

            # Update learning rate. Linearly interpolate from lr_start to lr_end
            for grp in optimizer.param_groups:
                grp['lr'] = training_lr_start + \
                    float(i / training_num_steps) * \
                    (training_lr_end - training_lr_start)

            if i % training_log_every == 0:
                logger.info(
                    f'Training step: {i}, le: {le.item()}, lo: {lo.item()}')
                e_logs.append((le.item(), lo.item()))

        model.eval()
        ######### End of training #########

        if (normalize_for_training):
            skinning_weight_function = NormalizedSkinningWeightsFcn(model, bb_min, bb_max)
        else:
            skinning_weight_function = SkinningWeightsFcn(model)

        return SimplicitsObject(pts, yms, prs, rhos, appx_vol, skinning_weight_function)

    @staticmethod
    def create_rigid(pts, yms, prs, rhos, appx_vol=1):
        r"""Creates a rigid SimplicitsObject with a single weight for affine deformations.

        This method creates a SimplicitsObject that behaves as a rigid body. At low stiffness values
        (young's modulus/ym), deformations will not be expressive, but with high stiffness values,
        the object will act as rigid.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float], optional): Approximate volume. Can be either:
                - A tensor of shape (1,)
                - A float value. Defaults to 1

        Returns:
            SimplicitsObject: A rigid SimplicitsObject with a constant weight function
        """
        def constant_weight_function(x):
            return torch.ones(
                x.shape[0], 1, device=x.device, dtype=x.dtype)

        return SimplicitsObject.create_from_function(pts, yms, prs, rhos, appx_vol, constant_weight_function)

    @staticmethod
    def create_from_function(pts, yms, prs, rhos, appx_vol, fcn):
        r"""Creates a SimplicitsObject with a custom skinning weight function.

        This method creates a SimplicitsObject using a user-provided function to compute skinning weights.
        The function should take points as input and return a matrix of skinning weights.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume. Can be either:
                - A tensor of shape (1,)
                - A float value
            fcn (callable): Function that takes points and returns skinning weights matrix

        Returns:
            SimplicitsObject: A SimplicitsObject with the provided skinning weight function
        """
        return SimplicitsObject(pts, yms, prs, rhos, appx_vol, skinning_weight_function=fcn)

    def __init__(self, pts, yms, prs, rhos, appx_vol, skinning_weight_function=None):
        r"""Initialize a SimplicitsObject with geometry, material properties, and skinning weights.

        A SimplicitsObject is a collection of points, material properties, and a linear blend skinning
        weight function that can be used to deform the object. Objects can be initialized in several ways
        using the static factory methods (read their docstrings for more details). Objects can also be
        denoted as kinematic or dynamic (default). Kinematic objects still have handles, but they are
        not solved for during simulation.

        Args:
            pts (torch.Tensor): Points tensor of shape (N, 3) representing the object's geometry
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density. Can be either:
                - A tensor of shape (N,) for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume. Can be either:
                - A tensor of shape (1,)
                - A float value
            skinning_weight_function (callable, optional): Function that takes points and returns skinning weights matrix.
                If None, the object will be rigid. Defaults to None

        """
        if not torch.is_tensor(yms):
            yms = torch.full((pts.shape[0],), yms, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(prs):
            prs = torch.full((pts.shape[0],), prs, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(rhos):
            rhos = torch.full((pts.shape[0],), rhos, dtype=pts.dtype, device=pts.device)
        if not torch.is_tensor(appx_vol):
            appx_vol = torch.tensor([appx_vol], dtype=pts.dtype, device=pts.device)

        self.pts = pts
        self.yms = yms
        self.prs = prs
        self.rhos = rhos
        self.appx_vol = appx_vol

        self.num_handles = skinning_weight_function(pts[:1]).shape[1]

        self.skinning_weight_function = skinning_weight_function


        self.device = pts.device
        self.dtype = pts.dtype

class SimulatedObject:
    def __init__(self, obj: SimplicitsObject, id: int, num_qp, num_cp, init_transform, is_kinematic=False):
        r""" Initialize a simulated object from a simplicits object. Users should avoid direct access to this class.
            Instead object-wise getters/setters for its parameters, forces and materials in the Scene to update/modify this object.

            Args:
                obj (SimplicitsObject): Saves the Simplicits Object to be used in the sim by reference here
                id (int): Object id
                num_qp (int, optional): Number of quadrature points (integration primitives). Defaults to 1000
                num_cp (int, optional): Number of collision points. Defaults to 1000. TODO: Not used yet.
                init_transform (torch.Tensor, optional): Initial transformation matrix of shape (12, 1). Defaults to None
        """

        self.simplicits_object = obj
        self.is_kinematic = is_kinematic
        self.id = id
        if num_qp >= self.simplicits_object.pts.shape[0]:
            num_qp = self.simplicits_object.pts.shape[0]
        self.num_qp = num_qp
        self.num_cp = num_cp
        self.init_transform = init_transform
        self.num_handles = obj.num_handles

        # Per-sampled primitive values
        self.sample_indices = None
        self.sample_pts = None
        self.sample_rhos = None
        self.sample_vols = None
        self.sample_yms = None
        self.sample_prs = None
        self.sample_masses = None

        # Per-sampled primitive sparse matrices
        self.sample_B = None
        self.sample_dFdz = None

        # Dofs
        self.z = None
        self.z_prev = None
        self.z_dot = None

        self._sample_cubatures()
        self.reset_sim_state()
        self._set_sim_constants()

    def __str__(self):
        r"""String describing object.

        Returns:
            string: String description of object
        """
        unique_densities = torch.unique(self.sample_rhos).tolist()
        unique_yms = torch.unique(self.sample_yms).tolist()
        unique_prs = torch.unique(self.sample_prs).tolist()
        appx_vol = self.simplicits_object.appx_vol
        return f"SimulatedObject(Densities={unique_densities}, Yms={unique_yms}, Prs={unique_prs}, AppxVol={appx_vol}, num_cub_pts={self.sample_pts.shape[0]}, kinematic={self.is_kinematic})"

    def _sample_cubatures(self):
        r"""Internal function to sample cubature points (and collision points and other per-primitive values) over the simplicits object.
        """

        # Use all if num_qp is less that self.simplicits_object.pts
        if self.num_qp < self.simplicits_object.pts.shape[0]:
            self.sample_indices = torch.randperm(
                len(self.simplicits_object.pts))[:self.num_qp]
        else:
            self.sample_indices = torch.arange(len(self.simplicits_object.pts))

        self.sample_pts = self.simplicits_object.pts[self.sample_indices, :].float()
        self.sample_rhos = self.simplicits_object.rhos[self.sample_indices].float()
        self.sample_vols = torch.full(
            (self.num_qp,), float(self.simplicits_object.appx_vol / self.num_qp),
            device=self.sample_pts.device, dtype=self.sample_pts.dtype)  # uniform integration sampling over vol
        self.sample_yms = self.simplicits_object.yms[self.sample_indices].float()
        self.sample_prs = self.simplicits_object.prs[self.sample_indices].float()
        self.sample_masses = (self.simplicits_object.appx_vol /
                              self.num_qp) * self.sample_rhos

        self.sample_skinning_weights = self.simplicits_object.skinning_weight_function(
            self.sample_pts)

    def _set_sim_constants(self):
        r""" Constants required for simulation. Computed upfront. Must be updated when _sample_cubature() is called
        """
        self.sample_B = sparse_lbs_matrix(
            wp.from_torch(self.sample_skinning_weights),
            wp.from_torch(self.sample_pts, dtype=wp.vec3))
        self.sample_B.nnz_sync()

        # Don't create this expensive autodiffed dFdz matrix for kinematic objects TODO: Make analytical dFdz 
        # But B matrix is cheap to compute, leave it alone
        if self.is_kinematic:
            num_rows = 9*self.sample_pts.shape[0]
            num_cols = 12*self.sample_skinning_weights.shape[1]
            self.sample_dFdz = wps.bsr_zeros(
                num_rows, num_cols, wp.float32)
        else:
            self.sample_dFdz = sparse_dFdz_matrix_from_dense(
                self.simplicits_object.skinning_weight_function, self.sample_pts)
        self.sample_dFdz.nnz_sync()
        
        self.sample_B_dense = warp_utilities.bsr_to_torch(
            self.sample_B).to_dense()
        self.sample_dFdz_dense = warp_utilities.bsr_to_torch(
            self.sample_dFdz).to_dense()

    def reset_sim_state(self):
        r"""Reset the simulation state. Object's handle transforms are set back to initial deformations.  
        This does not reset any material parameters or simplicits object parameters.
        """

        self.z = torch.zeros((self.num_handles) * 12,
                             dtype=self.simplicits_object.dtype, device=self.simplicits_object.device)
        # Sets the final (constant) handle 
        if self.init_transform is not None:
            self.z[-12:] = self.init_transform.flatten()
        
        self.z_prev = self.z.clone().detach()
        self.z_dot = torch.zeros_like(
            self.z, device=self.simplicits_object.device)


class SimplicitsScene:
    def __init__(self, device='cuda', dtype=torch.float, timestep=0.03, max_newton_steps=5, max_ls_steps=10):
        r"""Initializes a simplicits scene. SimplicitsObjects can be added to the scene. Scene forces such as floor and gravity can be set on the scene

        Args:
            device (str, optional): Defaults to 'cuda'.
            dtype (torch.dtype, optional): Defaults to torch.float.
            timestep (float, optional): Sim time-step. Defaults to 0.03.
            max_newton_steps (int, optional): Newton steps used in time integrator. Defaults to 20.
            max_ls_steps (int, optional): Line search steps used in time integrator. Defaults to 30.
        """

        self.device = device
        self.dtype = dtype

        self.direct_solve = True # default to true
        self.use_cuda_graphs = False # default to false

        self.timestep = timestep
        self.current_sim_step = 0

        self.max_newton_steps = max_newton_steps
        self.max_ls_steps = max_ls_steps
        self.newton_hessian_regularizer = 1e-4
        self.cg_tol = 1e-4
        self.cg_iters = 100
        self.conv_tol = 1e-4
        self.conv_criteria = 1

        self.current_id = 0
        self.sim_obj_dict = {}

        #### Simulation Constants ####
        # Render constants
        self.scene_B = None

        # Sparse Matrices
        self.sim_B = None       # Warp sparse LBS matrix
        self.sim_BMB = None     # Warp sparse reduced mass matrix
        self.sim_dFdz = None    # Warp sparse dFdz matrix
        self.sim_M = None       # Warp sparse diagonal mass matrix
        self.sim_Pt = None       # Warp sparse projection matrix transpose
        self.sim_P = None       # Warp sparse projection matrix

        # Per-primitive values
        self.sim_skinning_weights = None  # wp.array(num_qp, dtype=wp.vec3)
        self.sim_pts = None          # wp.array(num_qp, dtype=wp.vec3)
        self.sim_pts_flat = None     # wp.array(3*num_qp, dtype=wp.float32)
        self.sim_yms = None          # wp.array(num_qp, dtype=wp.float32)
        self.sim_prs = None          # wp.array(num_qp, dtype=wp.float32)
        self.sim_mus = None          # wp.array(num_qp, dtype=wp.float32)
        self.sim_lams = None         # wp.array(num_qp, dtype=wp.float32)
        self.sim_rhos = None         # wp.array(num_qp, dtype=wp.float32)
        self.sim_vols = None         # wp.array(num_qp, dtype=wp.float32)
        self.sim_masses = None       # wp.array(num_qp, dtype=wp.float32)

        # DOFs
        self.sim_z = None         # wp.array(num_handles*12, dtype=wp.float32)
        self.sim_z_prev = None    # wp.array(num_handles*12, dtype=wp.float32)
        self.sim_z_dot = None     # wp.array(num_handles*12, dtype=wp.float32)

        # Generated kernels
        self.force_dict = {
            "pt_wise": {},
            "defo_grad_wise": {}
        }

        #Structure the force_dict like this.
        #force_dict = {
        #    "pt_wise":{
        #        "gravity":{
        #            "object": gravity_struct,
        #            "energy": gravity_energy_fcn,
        #            "gradient": gravity_gradient_fcn,
        #            "hessian": gravity_hessian_fcn,
        #        },
        #        "name_of_bc":{
        #            "object": bc_struct,
        #            "energy": bc_energy_fcn,
        #            "gradient": bc_gradient_fcn,
        #            "hessian": bc_hessian_fcn,
        #        }
        #    },
        #    "defo_grad_wise":{
        #        "material":{
        #            "object": material_struct,
        #            "energy": material_energy_fcn,
        #            "gradient": material_gradient_fcn,
        #            "hessian": material_hessian_fcn,
        #        }
        #    }
        #}

        self.partial_wp_newton_E = None
        self.partial_wp_newton_G = None
        self.partial_wp_newton_H = None

        self.object_to_z_map = None
        self.object_to_qp_map = None
        self.qp_to_object_map = None
        self.z_to_object_map = None

        self.kin_obj_list = None
        self.kin_obj_to_z_map = None
        self.kin_obj_to_qp_map = None
        self.qp_is_kinematic = None

        self._ready_for_forces = False
        self._ready_for_sim = False

    def _compute_sim_constants(self):
        _num_qp = []
        _num_cp = []
        _num_handles = []

        _stacked_pts = []
        _stacked_rhos = []

        _stacked_yms = []
        _stacked_prs = []
        _stacked_masses = []
        _stacked_vols = []
        _stacked_skinning_weights = []
        _stacked_skinning_weight_grads = []

        _stacked_sparse_B = []
        _stacked_sparse_dFdz = []

        _object_wise_jacobians = []

        _qp_to_object_map = []  # primitive index, object id value
        _object_to_qp_map = {}  # key is object id, value is array of qp indices
        _z_to_object_map = []  # key is z index, value is object id
        _object_to_z_map = {}  # dof index, object id value
        _kin_obj_list = []     # list of kinematic object ids
        _kin_obj_to_z_map = {}  # dof index, object id value
        _kin_obj_to_qp_map = {}  # qp index, object id value

        z_index = 0
        x_index = 0
        for object in self.sim_obj_dict.values():
            _num_qp.append(object.num_qp)
            _num_cp.append(object.num_cp)
            _num_handles.append(object.num_handles)

            # Mapping from object to z and x indices
            _object_to_z_map[object.id] = wp.array(
                np.arange(z_index, z_index + object.num_handles * 12), dtype=wp.int32)
            _object_to_qp_map[object.id] = wp.array(
                np.arange(x_index, x_index + object.num_qp), dtype=wp.int32)
            # Add torch array full of object id to list
            _qp_to_object_map.append(torch.full(
                (object.num_qp,), object.id, dtype=torch.int32, device=self.device))
            _z_to_object_map.append(torch.full(
                (object.num_handles * 12,), object.id, dtype=torch.int32))

            if object.is_kinematic:
                _kin_obj_list.append(object.id)
                _kin_obj_to_z_map[object.id] = wp.array(
                    np.arange(z_index, z_index + object.num_handles * 12), dtype=wp.int32)
                _kin_obj_to_qp_map[object.id] = wp.array(
                    np.arange(x_index, x_index + object.num_qp), dtype=wp.int32)

            z_index += object.num_handles * 12
            x_index += object.num_qp

            _stacked_pts.append(object.sample_pts)
            _stacked_rhos.append(object.sample_rhos)
            _stacked_vols.append(object.sample_vols)
            _stacked_masses.append(object.sample_masses)
            _stacked_yms.append(object.sample_yms)
            _stacked_prs.append(object.sample_prs)
            _stacked_sparse_B.append(object.sample_B)
            _stacked_sparse_dFdz.append(object.sample_dFdz)
            _stacked_skinning_weights.append(object.sample_skinning_weights)
        
        self.object_to_z_map = _object_to_z_map
        self.object_to_qp_map = _object_to_qp_map
        self.qp_to_object_map = wp.from_torch(torch.cat(_qp_to_object_map))
        self.z_to_object_map = wp.from_torch(torch.cat(_z_to_object_map))
        self.kin_obj_list = _kin_obj_list
        self.kin_obj_to_z_map = _kin_obj_to_z_map
        self.kin_obj_to_qp_map = _kin_obj_to_qp_map
        # Use the maps above to get a list of whether all qp indices are kinematic
        t_qp_is_kinematic = torch.zeros(
            sum(_num_qp), dtype=torch.int32, device=self.device)
        for kin_id in _kin_obj_list:
            t_qp_is_kinematic[wp.to_torch(self.kin_obj_to_qp_map[kin_id])] = 1
        self.qp_is_kinematic = wp.from_torch(t_qp_is_kinematic)

        # Other scene constants
        self.sim_skinning_weights = wp.from_torch(
            torch.block_diag(*_stacked_skinning_weights).contiguous())
        self.sim_pts = wp.from_torch(
            torch.cat(_stacked_pts, dim=0).contiguous(), dtype=wp.vec3)
        self.sim_pts_flat = wp.array(self.sim_pts, dtype=wp.float32).flatten()

        mus, lams = to_lame(
            torch.cat(_stacked_yms, dim=0).contiguous(),
            torch.cat(_stacked_prs, dim=0).contiguous())
        self.sim_mus = wp.from_torch(mus)
        self.sim_lams = wp.from_torch(lams)
        self.sim_yms = wp.from_torch(
            torch.cat(_stacked_yms, dim=0).contiguous())
        self.sim_prs = wp.from_torch(
            torch.cat(_stacked_prs, dim=0).contiguous())

        self.sim_rhos = wp.from_torch(
            torch.cat(_stacked_rhos, dim=0).contiguous())
        self.sim_vols = wp.from_torch(
            torch.cat(_stacked_vols, dim=0).contiguous())

        ###################################################
        # Scene LBS Matrix
        self._dof_bs = 4

        # Create the projection matrix that projects out the kinematic dofs
        # Get list of all dofs corresponding to kinematic objects
        kin_dofs = [wp.to_torch(self.kin_obj_to_z_map[kin_id])
                    for kin_id in _kin_obj_list]
        # Pass in the total number of dofs and the list of kinematic dofs
        if len(kin_dofs) > 0:
            temp_sim_Pt = warp_utilities.warp_csr_from_torch_dense(torch_utilities.create_projection_matrix(
                12*sum(_num_handles), torch.cat(kin_dofs, dim=0)))
            self.sim_Pt = wps.bsr_copy(
                temp_sim_Pt, block_shape=(1, self._dof_bs))
            self.sim_P = self.sim_Pt.transpose()
        else:
            self.sim_Pt = None
            self.sim_P = None

        # Mass Matrix
        wp_sim_masses = wp.from_torch(
            torch.cat(_stacked_masses, dim=0).contiguous())  # [scene_pts, 1]
        np_sim_masses = np.repeat(wp_sim_masses.numpy(), 3)
        # [3 * scene_pts, 3 * scene_pts] sparse diagonal matrix
        self.sim_M = wps.bsr_diag(wp.array(np_sim_masses, dtype=wp.float32))

        # LBS matrix and Simplicits mass matrix
        temp_B = warp_utilities.block_diagonalize(_stacked_sparse_B)
        self.sim_B = wps.bsr_copy(temp_B, block_shape=(1, self._dof_bs))

        self.sim_BMB = wps.bsr_transposed(self.sim_B)@self.sim_M@self.sim_B

        # Scene dFdz Matrix
        temp_dFdz = warp_utilities.block_diagonalize(
            _stacked_sparse_dFdz)
        self.sim_dFdz = wps.bsr_copy(temp_dFdz, block_shape=(9, self._dof_bs))

        # Syncs host-side with device-side nnz
        self.sim_M.nnz_sync()
        self.sim_B.nnz_sync()
        self.sim_dFdz.nnz_sync()
        self.sim_BMB.nnz_sync()

        elastic_struct = NeohookeanElasticMaterial(
            mu=self.sim_mus,
            lam=self.sim_lams,
            integration_pt_volume=self.sim_vols
        )

        self.force_dict["defo_grad_wise"]["material"] = {}
        self.force_dict["defo_grad_wise"]["material"]["object"] = elastic_struct
        self.force_dict["defo_grad_wise"]["material"]["coeff"] = 1.0

    def _create_sim_variables(self):
        self.reset_scene()

        self._energy_graph = None
        self._gradient_graph = None
        self._scene_energy = wp.empty(2, dtype=float)
        self._scene_gradient = wp.empty_like(self.sim_z)
        self._eval_dx = wp.empty_like(self.sim_pts)
        self._eval_z = wp.empty_like(self.sim_z)
        self._eval_delta_dz = wp.empty_like(self.sim_z)

    def set_object_initial_transform(self, object_id, init_transform):
        r"""Sets the initial transform of an object.

        .. note::
           this method reset the scene

        Args:
            object_id (int): Id of the object to set the initial transform of
            init_transform (torch.Tensor):
                3x4 torch tensor specifying object's initial skinning transform. 
                This argument takes a standard transformation, not a delta. 
                Subsequently, the Identity matrix is subtracted from it and the delta transform is saved.
        """
        if self.current_sim_step > 0:
            raise ValueError("Cannot set initial transform after simulation has started. Reset the scene first using scene.reset_scene()")
        
        obj = self.sim_obj_dict[object_id]
        
        if obj.is_kinematic:
            raise ValueError("Object is a kinematic object, set its constant transformation using set_kinematic_object_transform(...)")
        
        if not torch.is_tensor(init_transform):
            raise ValueError("init_transform must be a torch.Tensor")
        
        relative_transform = torch_utilities.standard_transform_to_relative(init_transform)

        obj.init_transform = relative_transform
        self.reset_scene() # quick way to update the scene's dofs

    def get_object_transforms(self, object_id):
        """
        Returns the current 4x4 padded standard transforms of the object's handles
        
        Args:
            object_id (int): Id of the object to get the transforms of

        Returns:
            torch.tensor: num_handles x 4 x 4 torch tensor
        """
        tfms = None
        
        # Sets tfms to the current objects transforms if sim_z is set, 
        # otherwise sets it to the initial transforms (both are 3x4 tensors)
        if self.sim_z is not None:
            wp_tfms = wp.clone(self.sim_z[self.object_to_z_map[object_id]])
            tfms = wp.to_torch(wp_tfms, requires_grad=False).reshape(
                (-1, 3, 4))
        else:
            # Sets to the initial transforms
            tfms = torch.zeros(self.sim_obj_dict[object_id].num_handles, 3, 4, device=self.device, dtype=self.dtype) 
            tfms[-1] = self.sim_obj_dict[object_id].init_transform.reshape(-1, 3, 4) 

        # Pad with 0,0,0,1 rows to make each transform 4x4
        padding = torch.zeros(tfms.shape[0], 1, 4, device=self.device, dtype=self.dtype)
        padding[:, 0, 3] = 1.0
        
        # Concatenate the padding to the bottom of each transform
        padded_tfms = torch.cat([tfms, padding], dim=1)
        return padded_tfms

    def add_object(self, sim_object: SimplicitsObject, num_qp=1000, num_cp=1000, init_transform = None, is_kinematic=False):
        r"""Adds a simplicits object to the scene as a SimulatedObject.

        Args:
            sim_object (SimplicitsObject): Simplicits object that will be wrapped into a SimulatedObject for this scene.
            num_qp (float): Number of quadrature points (sample points to integrate over)
            num_cp (float): # TODO (Clement): REMOVED IN ANOTHER MR
            init_transform (torch.Tensor): 3x4 or 4x4 torch tensor specifying object's initial skinning transform. 
                                            This argument takes a standard transformation, not a delta. 
                                            Subsequently, the Identity matrix is subtracted from it and the delta transform is saved.
            is_kinematic (bool): Object is kinematic if it is not solved for during dynamics simulation.
        """
        if self._ready_for_forces:
            raise RuntimeError("Cannot object after a force is set, please create a new scene")

        # Check if init transform is a 3x4 or 4x4 tensor, convert to 3x4 if necessary
        # Subtract identity transform from init transform to get relative transform  
        if torch.is_tensor(init_transform):
            relative_transform = torch_utilities.standard_transform_to_relative(init_transform)
        else:
            relative_transform = torch.zeros(3, 4, device=self.device, dtype=self.dtype)
            
        self.sim_obj_dict[self.current_id] = SimulatedObject(
            sim_object, self.current_id, num_qp, num_cp, relative_transform, is_kinematic)
            
        self.current_id += 1
        return self.current_id - 1
    
    def set_kinematic_object_transform(self, obj_idx, transform):
        r"""Sets the transform of a kinematic object. This can be done during simulation to script the kinematic object's motion.
        
        Args:
            obj_idx (int): Id of object
            transform (torch.Tensor): 3x4 or 4x4 torch tensor specifying object's transform.
        """
        if not torch.is_tensor(transform):
            raise ValueError("transform must be a torch.Tensor")
        
        t_sim_z = wp.to_torch(self.sim_z)
        obj = self.get_object(obj_idx)
        if not obj.is_kinematic:
            raise ValueError("Object is not a kinematic object. Use set_object_initial_transform(...) to set the initial transform of a non-kinematic object.")
        
        obj.init_transform = torch_utilities.standard_transform_to_relative(transform)
        obj.reset_sim_state()
        t_sim_z[wp.to_torch(self.object_to_z_map[obj_idx])] = obj.z.flatten()
        
        self.sim_z = wp.from_torch(t_sim_z)

    def set_scene_gravity(self, acc_gravity=torch.tensor([0, 9.8, 0]), gravity_coeff=1.0):
        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()
        wp_g_vec = wp.vec3(acc_gravity[0], acc_gravity[1], acc_gravity[2])

        # Gravity
        gravity_struct = Gravity(
            g=wp_g_vec,
            integration_pt_density=self.sim_rhos,
            integration_pt_volume=self.sim_vols
        )

        self.force_dict["pt_wise"]["gravity"] = {}
        self.force_dict["pt_wise"]["gravity"]["object"] = gravity_struct
        self.force_dict["pt_wise"]["gravity"]["coeff"] = gravity_coeff

    def set_scene_floor(self, floor_height=0.0, floor_axis=1, floor_penalty=10000.0, flip_floor=False):
        r"""Sets the floor in the scene. Applies it to all objects in scene.

        Args:
            floor_height (float, optional): Floor height. Defaults to 0.
            floor_axis (int, optional): Direction of floor. 0 is x, 1 is y, 2 is z. Defaults to 1.
            floor_penalty (float, optional): Stiffness of floor. Defaults to 10000.
        """
        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()

        floor_struct = Floor(
            floor_height=floor_height,
            floor_axis=floor_axis,
            flip_floor=int(flip_floor),
            # set integ vol to constant ones
            integration_pt_volume=wp.ones_like(self.sim_vols)
        )

        self.force_dict["pt_wise"]["floor"] = {}
        self.force_dict["pt_wise"]["floor"]["object"] = floor_struct
        self.force_dict["pt_wise"]["floor"]["coeff"] = floor_penalty

    # TODO(cfujitsang)
    #def set_object_materials(self, obj_idx, yms=None, prs=None, rhos=None, elasticity_type="neohookean"):
    #    r"""Sets object's material properties

    #    Args:
    #        obj_idx (int): Id of object
    #        yms (torch.Tensor, optional): Sets Yms of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
    #        prs (torch.Tensor, optional): Sets Prs of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
    #        rhos (torch.Tensor, optional): Sets densities of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
    #    """
    #    if not self._ready_for_forces:
    #        self._get_scene_ready_for_forces()

    #    simulated_object = self.sim_obj_dict[obj_idx]
    #    if yms is not None:
    #        simulated_object.sample_yms = torch.full(
    #            (simulated_object.num_qp,), yms, device=self.device, dtype=self.dtype)
    #    if prs is not None:
    #        simulated_object.sample_prs = torch.full(
    #            (simulated_object.num_qp,), prs, device=self.device, dtype=self.dtype)
    #    if rhos is not None:
    #        simulated_object.sample_rhos = torch.full(
    #            (simulated_object.num_qp,), rhos, device=self.device, dtype=self.dtype)
    #        simulated_object.sample_masses = (simulated_object.simplicits_object.appx_vol /
    #                                          simulated_object.num_qp) * simulated_object.sample_rhos
    #    self._ready_for_sim = False

    def set_object_boundary_condition(self, obj_idx, name, fcn, bdry_penalty=10000.0, pinned_x=None):
        r"""Sets boundary condition for object in scene

        Args:
            obj_idx (int): Id of object
            name (str): Boundary condition name
            fcn (Callable): Function that defines which indices the boundary condition applies to. fcn should return a boolean :math:`(n)` vector where bdry indices are 1.
            bdry_penalty (float): Boundary condition penalty coefficient.
        """

        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()

        boundary_struct = Boundary(
            integration_pt_volume=self.sim_vols)

        obj_global_indices = wp.to_torch(self.object_to_qp_map[obj_idx])
        deformed_pts = self.get_object_deformed_pts(obj_idx)
        # Get indices of points that are less than -0.4 in the x direction
        bdry_indx = torch.nonzero(
            fcn(deformed_pts), as_tuple=False).squeeze(1)  # squeeze dim 1 to get 1D tensor even if there is only one element
        bdry_pos = deformed_pts[bdry_indx]
        global_bdry_indx = obj_global_indices[bdry_indx]

        if pinned_x is None:
            pinned_x = bdry_pos

        boundary_struct.set_pinned(indices=wp.from_torch(
            global_bdry_indx, dtype=wp.int32), pinned_x=wp.from_torch(pinned_x, dtype=wp.vec3))

        self.force_dict["pt_wise"][name] = {}
        self.force_dict["pt_wise"][name]["object"] = boundary_struct
        self.force_dict["pt_wise"][name]["coeff"] = bdry_penalty

        return pinned_x

    def enable_collisions(self, collision_particle_radius=0.1,
                            detection_ratio=1.5,
                            impenetrable_barrier_ratio=0.25,
                            ignore_self_collision_ratio=100000.0, 
                            collision_penalty=1000.0, 
                            max_contact_pairs=10000,
                            friction=0.5):
        r"""Sets collision for object in scene

        Args:
            collision_particle_radius (float): Radius of the collision particle at which penalty begins to apply.
            detection_ratio (float): Collision detection radius described as a ratio relative to the collision_particle_radius.
            impenetrable_barrier_ratio (float): Collision barrier radius described as a ratio relative to the collision_particle_radius.
            ignore_self_collision_ratio (float): Collision immune radius described as a ratio relative to the collision_particle_radius.
            collision_penalty: Controls the stiffness of the collision interaction.
            max_contact_pairs: Maximum number of contact pairs to detect. If this is too low, some contacts may be missed. If this is too high, memory may run out/jacobian may be too large.
        """

        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()

        collision_struct = Collision(
            dt=self.timestep,
            collision_particle_radius=collision_particle_radius,
            detection_ratio=detection_ratio,
            impenetrable_barrier_ratio=impenetrable_barrier_ratio,
            ignore_self_collision_ratio=ignore_self_collision_ratio,
            collision_penalty_stiffness=collision_penalty,
            friction_regularization=0.1, # Don't expose to users, its fine
            friction_fluid=0.1, # Don't expose to users, its fine
            friction=friction,
            max_contacting_pairs=max_contact_pairs,
            bounds=True
        )

        self.force_dict["collision"] = {}
        self.force_dict["collision"]["object"] = collision_struct
        self.force_dict["collision"]["coeff"] = collision_penalty

        self.detect_collision(self.sim_z)

    def detect_collision(self, z):
        r"""Resets the collision jacobian when new contact pairs are found."""
        assert z.shape[0] == self.sim_B.shape[1]
        # TODO (Clement): change this to ["special"]["collision"] upon merge
        if "collision" not in self.force_dict or self.force_dict["collision"]["object"] is None:
            if len(self.sim_obj_dict) > 1:
                warnings.warn(
                    "Collision not enabled in scene with multiple objects", UserWarning)
            return

        collision_struct = self.force_dict["collision"]["object"]

        # Sets the collision points dx at the start of the timestep
        # TODO: Move this to a separate function if detect_collisions is called multiple times in a timestep
        dx0 = wp.array((self.sim_B@z), dtype=wp.vec3)
        collision_struct.cp_dx_at_nm_iteration_0 = dx0
        #-----------------------------------------------------------
        
        # Detecting collisions
        dx = wp.array((self.sim_B@z), dtype=wp.vec3)
        collision_struct.detect_collisions(cp_dx=dx,  # TODO: Set this to CP_dx once we have different cps than qps
                                           cp_x0=self.sim_pts,
                                           cp_obj_ids=self.qp_to_object_map,
                                           cp_is_static=None)
        
        # Builds collision jacobian    
        collision_struct.build_jacobian(
            cp_w=self.sim_skinning_weights, cp_x0=self.sim_pts, cp_is_static=self.qp_is_kinematic)

    def build_preconditioner(self, lhs):
        return warp_utilities.build_preconditioner(lhs)

    def compute_collision_bounds(self, dz, z):
        if "collision" not in self.force_dict or self.force_dict["collision"]["object"] is None:
            return None

        assert dz.shape[0] == self.sim_B.shape[1]

        collision_struct = self.force_dict["collision"]["object"]
        if collision_struct.num_contacts == 0:
            return None

        dx = wp.array((self.sim_B@z), dtype=wp.vec3)

        delta_dx = wp.array((self.sim_B@dz), dtype=wp.vec3)

        wp_bounds = collision_struct.compute_bounds(cp_delta_dx=delta_dx,  # B*z_k, z_k = z at nm step 0
                                                    cp_dx=dx,  # B*z
                                                    cp_x0=self.sim_pts)  # x0

        return wp_bounds

    def reset_scene(self):
        r"""Resets the scene and all objects in it.
        """
        self.current_sim_step = 0
        # Reset each object in scene
        _stacked_z = []
        for object in self.sim_obj_dict.values():
            object.reset_sim_state()
            _stacked_z.append(object.z)
        z = torch.cat(_stacked_z, dim=0).contiguous()
        assert z.shape[0] == self.sim_B.shape[1]

        self.sim_z = wp.from_torch(z.flatten())
        self.sim_z_prev = wp.zeros_like(self.sim_z)
        self.sim_z_dot = wp.zeros_like(self.sim_z)

    def _assemble_energies(self, z, delta_dz):
        x0 = self.sim_pts

        # copy to fixed memory location so we can use in graph
        wp.copy(src=z, dest=self._eval_z)
        wp.copy(src=delta_dz, dest=self._eval_delta_dz)

        def eval_fixed_energies():
            # Names of gradients to assemble
            pt_names = list(self.force_dict["pt_wise"].keys())
            defo_grad_names = list(self.force_dict["defo_grad_wise"].keys())

            # Get grad function input variables
            F_ele = compute_defo_grad(
                self._eval_z, self.sim_dFdz)  # wp.array((n,), dtype=wp.mat33)

            wps.bsr_mv(A=self.sim_B, x=self._eval_z, y=self._eval_dx)

            self._scene_energy.zero_()

            for e in pt_names:
                force_obj = self.force_dict["pt_wise"][e]["object"]
                coeff = self.force_dict["pt_wise"][e]["coeff"]
                force_obj.energy(self._eval_dx, x0, coeff, self._scene_energy)

            for e in defo_grad_names:
                force_obj = self.force_dict["defo_grad_wise"][e]["object"]
                coeff = self.force_dict["defo_grad_wise"][e]["coeff"]
                force_obj.energy(F_ele, coeff, self._scene_energy)

            # Kinetic energy
            BMBz = wp.array(self.sim_BMB @ self._eval_delta_dz,
                            dtype=float).flatten()
            wp.utils.array_inner(self._eval_delta_dz, BMBz,
                                 out=self._scene_energy[1:])
        if self.use_cuda_graphs:
            if self._energy_graph is None:
                eval_fixed_energies()  # dry-run to force load all the modules required for energy eval
                with wp.ScopedCapture(force_module_load=False) as capture:
                    eval_fixed_energies()

                self._energy_graph = capture.graph
            wp.capture_launch(self._energy_graph)
        else:
            eval_fixed_energies()

        # Special case for collision energy
        # Since we might use different cps than qps
        if "collision" in self.force_dict and self.force_dict["collision"]["object"].num_contacts > 0:
            self.force_dict["collision"]["object"].energy(
                self._eval_dx, x0, self.force_dict["collision"]["coeff"], self._scene_energy)

        energies = self._scene_energy.numpy()
        pe_sum = energies[0]
        ke = energies[1] * 0.5
        return pe_sum, ke

    def _assemble_gradients(self, z):
        # Steps
        # 1. Get global gradients
        # 2. Loop through objects and do J.T * grad[obj_inds] in torch
        # 3. Concatenate the gradients

        # copy to fixed memory location so we can use in graph
        wp.copy(src=z, dest=self._eval_z)

        def eval_fixed_gradients():
            num_pts = int(self.sim_B.shape[0]/3)

            # Names of gradients to assemble
            pt_names = list(self.force_dict["pt_wise"].keys())
            defo_grad_names = list(self.force_dict["defo_grad_wise"].keys())

            # Get grad function input variables
            F_ele = compute_defo_grad(
                self._eval_z, self.sim_dFdz)  # wp.array((n,), dtype=wp.mat33)

            wps.bsr_mv(A=self.sim_B, x=self._eval_z, y=self._eval_dx)

            self._scene_gradient.zero_()
            x0 = self.sim_pts

            # Get global pt-wise gradients
            scene_dEdx = wp.zeros(num_pts, dtype=wp.vec3)
            for e in pt_names:
                force_obj = self.force_dict["pt_wise"][e]["object"]
                coeff = self.force_dict["pt_wise"][e]["coeff"]
                force_obj.gradient(self._eval_dx, x0, coeff, scene_dEdx)

            # Get global defo-grad gradients
            scene_dEdF = wp.zeros(num_pts, dtype=wp.mat33)
            for e in defo_grad_names:
                force_obj = self.force_dict["defo_grad_wise"][e]["object"]
                coeff = self.force_dict["defo_grad_wise"][e]["coeff"]
                force_obj.gradient(F_ele, coeff, scene_dEdF)

            wps.bsr_mv(A=self.sim_B, x=scene_dEdx,
                       y=self._scene_gradient, beta=1.0, transpose=True)
            wps.bsr_mv(A=self.sim_dFdz, x=scene_dEdF,
                       y=self._scene_gradient, beta=1.0, transpose=True)

        if self.use_cuda_graphs:
            if self._gradient_graph is None:
                with wp.ScopedCapture(force_module_load=False) as capture:
                    eval_fixed_gradients()

                self._gradient_graph = capture.graph
            wp.capture_launch(self._gradient_graph)
        else:
            eval_fixed_gradients()

        #### COLLISIONS ####
        # Special case for collision gradient since we might use different cps than qps
        if "collision" in self.force_dict and self.force_dict["collision"]["object"].num_contacts > 0:
            collision_struct = self.force_dict["collision"]["object"]
            collision_coeff = self.force_dict["collision"]["coeff"]
            collision_dEdx = collision_struct.gradient(
                self._eval_dx, self.sim_pts, collision_coeff)

            wps.bsr_mv(A=collision_struct.collision_J, x=collision_dEdx, y=self._scene_gradient,
                       beta=1.0, transpose=True)

        return self._scene_gradient

    def _assemble_hessians(self, z):
        # Steps
        # 1. Get global hessians
        # 2. Loop through objects and do J.T * hess[obj_inds] @ J in warp/torch
        # 3. Concatenate the hessians into a big list [(i,i,H_ii), (i,j,H_ij), ....]

        num_pts = int(self.sim_B.shape[0]/3)
        num_dofs = self.sim_B.shape[1]
        # Names of gradients to assemble
        pt_names = list(self.force_dict["pt_wise"].keys())
        defo_grad_names = list(self.force_dict["defo_grad_wise"].keys())

        # Get grad function input variables
        F_ele = compute_defo_grad(
            z, self.sim_dFdz)
        dx = wp.array(self.sim_B@wp.array(z, dtype=wp.vec3), dtype=wp.vec3)
        x0 = self.sim_pts

        # Get global pt-wise hessians in torch
        scene_d2Edx2 = torch.zeros(
            num_pts, 3, 3, device=self.device, dtype=self.dtype)
        for e in pt_names:
            force_obj = self.force_dict["pt_wise"][e]["object"]
            coeff = self.force_dict["pt_wise"][e]["coeff"]
            hess = force_obj.hessian(dx, x0, coeff)
            scene_d2Edx2 += wp.to_torch(hess, requires_grad=False)

        # Get global defo-grad hessians
        scene_d2EdF2 = torch.zeros(
            num_pts, 9, 9, device=self.device, dtype=self.dtype)
        for e in defo_grad_names:
            force_obj = self.force_dict["defo_grad_wise"][e]["object"]
            coeff = self.force_dict["defo_grad_wise"][e]["coeff"]
            hess = force_obj.hessian(F_ele, coeff)
            scene_d2EdF2 += wp.to_torch(hess, requires_grad=False)

        # Assemble the Hessians into a list of (i,j,H_ij)
        hess_list = []
        for obj in self.sim_obj_dict.values():
            qp_i = wp.to_torch(self.object_to_qp_map[obj.id])

            H_ii = torch_utilities.hess_reduction(obj.sample_B_dense, scene_d2Edx2[qp_i, :, :]) + torch_utilities.hess_reduction(
                obj.sample_dFdz_dense, scene_d2EdF2[qp_i, :, :])

            hess_list.append((obj.id, obj.id, H_ii))

        # this is a sparse matrix
        # H = warp_utilities.assemble_global_hessian(
        #     hess_list, self.object_to_z_map, z)

        #### COLLISIONS ####
        # Special case for collision hessian since we might use different cps than qps
        if "collision" in self.force_dict and self.force_dict["collision"]["object"].num_contacts > 0:
            collision_struct = self.force_dict["collision"]["object"]
            collision_coeff = self.force_dict["collision"]["coeff"]
            collision_hess = collision_struct.hessian(
                dx, x0, collision_coeff)  # (n, 3, 3)

            collision_hess = wp.to_torch(
                collision_hess, requires_grad=False)  # (n, 3, 3)
            obj_dofs = self.object_to_z_map

            for i, j in collision_struct.object_pairs:

                J_i = collision_struct.collision_J_dense[:, wp.to_torch(
                    obj_dofs[i])]
                J_j = collision_struct.collision_J_dense[:, wp.to_torch(
                    obj_dofs[j])]

                H_ij = torch_utilities.hess_reduction(J_i, collision_hess, J_j)
                hess_list.append((i, j, H_ij))

            # cJ = collision_struct.collision_J
            # cJt = collision_struct.collision_Jt
            # cH = warp_utilities.wp_hessian_reduction(cJt, collision_hess, cJ)
            # cH33 = wps.bsr_copy(cH, block_shape=(3, 3))
            # wps.bsr_axpy(cH33, H, alpha=1.0, beta=1.0)

        # this is a sparse matrix
        H = warp_utilities.assemble_global_hessian(
            hess_list, self.object_to_z_map, z, block_size=self._dof_bs)

        return H

    @wp.kernel
    def _displacement_delta_kernel(
        dt: float,
        z: wp.array(dtype=float),
        z_prev: wp.array(dtype=float),
        z_dot: wp.array(dtype=float),
        delta_dz: wp.array(dtype=float),
    ):
        i = wp.tid()
        delta_dz[i] = (z[i] - z_prev[i]) - dt*z_dot[i]

    @staticmethod
    def _displacement_delta(wp_z, wp_z_prev, wp_z_dot, dt):
        r"""Timestep displacement update, to use in inertia computations

        Args:
            wp_z (wp.array): Transforms
            wp_z_prev (wp.array): Previous transforms
            wp_z_dot (wp.array): Time derivative of transforms
            dt (float): Timestep
        """

        delta_dz = wp.empty_like(wp_z)
        wp.launch(SimplicitsScene._displacement_delta_kernel, dim=delta_dz.shape, inputs=[
                  dt, wp_z, wp_z_prev, wp_z_dot], outputs=[delta_dz])
        return delta_dz

    def _newton_E(self, wp_z, wp_z_prev, wp_z_dot,
                  wp_B, wp_BMB, dt, wp_dFdz, defo_grad_energies=None, pt_wise_energies=None):
        r"""Backward's euler energy used in newton's method

        Args:
            wp_z (wp.array): Transforms
            wp_z_prev (wp.array): Previous transforms
            wp_z_dot (wp.array): Time derivative of transforms
            wp_B (wp.sparse.bsr_matrix): Precomputed Jacobian, dx/dz
            wp_BMB (wp.sparse.bsr_matrix): Precomputed z-wise mass matrix.
            dt (float): Timestep
            wp_dFdz (wp.sparse.bsr_matrix): Precomputed jacobian
            defo_grad_energies (list, optional): List of energies. Defaults to [].
            pt_wise_energies (list, optional): List of energies. Defaults to [].

        Returns:
            float: Backward's euler energy scalar.
        """
        assert wp_z.shape[0] == wp_B.shape[1]
        wp_delta_dz = self._displacement_delta(wp_z, wp_z_prev, wp_z_dot, dt)
        pe_sum, ke = self._assemble_energies(wp_z, wp_delta_dz)

        wp_newton_energy = ke + dt*dt * pe_sum
        return wp_newton_energy

    def _newton_G(self, wp_z, wp_z_prev, wp_z_dot, wp_B, wp_BMB, dt, wp_dFdz, defo_grad_gradients=None, pt_wise_gradients=None):
        r"""Backward's euler gradient used in newton's method

        Args:
            wp_z (wp.array): Transforms
            wp_z_prev (wp.array): Previous transforms
            wp_z_dot (wp.array): Time derivative of transforms
            wp_B (wp.sparse.bsr_matrix): Precomputed Jacobian, dx/dz
            wp_BMB (wp.sparse.bsr_matrix): Precomputed z-wise mass matrix.
            dt (float): Timestep
            wp_dFdz (wp.sparse.bsr_matrix): Precomputed jacobian
            defo_grad_energies (list, optional): List of energies. Defaults to [].
            pt_wise_energies (list, optional): List of energies. Defaults to [].

        Returns:
            wp.array: Backward's euler gradient.
        """
        assert wp_z.shape[0] == wp_B.shape[1]

        newton_gradient = self._assemble_gradients(wp_z)

        wp_delta_dz = self._displacement_delta(wp_z, wp_z_prev, wp_z_dot, dt)
        wps.bsr_mv(wp_BMB, x=wp_delta_dz,
                   y=newton_gradient, alpha=1.0, beta=dt*dt)

        return newton_gradient

    def _newton_H(self, wp_z, wp_z_prev, wp_z_dot, wp_B, wp_BMB, dt, wp_dFdz, defo_grad_hessians=None, pt_wise_hessians=None):
        r"""Backward's euler hessian used in newton's method

        Args:
            wp_z (wp.array): Transforms
            wp_z_prev (wp.array): Previous transforms
            wp_z_dot (wp.array): Time derivative of transforms
            wp_B (wp.sparse.bsr_matrix): Precomputed Jacobian, dx/dz
            wp_BMB (wp.sparse.bsr_matrix): Precomputed z-wise mass matrix.
            dt (float): Timestep
            wp_dFdz (wp.sparse.bsr_matrix): Precomputed jacobian
            defo_grad_energies (list, optional): List of energies. Defaults to [].
            pt_wise_energies (list, optional): List of energies. Defaults to [].

        Returns:
            wp.sparse.bsr_matrix: Backward's euler hessian.
        """
        assert wp_z.shape[0] == wp_B.shape[1]
        newton_hessian = self._assemble_hessians(wp_z)

        # add mass matrix
        wps.bsr_axpy(wp_BMB, newton_hessian, alpha=1.0, beta=dt*dt)

        # add regularizer
        id_regularizer = wps.bsr_identity(
            newton_hessian.nrow,
            block_type=newton_hessian.dtype, device=wp_z.device)
        wps.bsr_axpy(id_regularizer, newton_hessian, alpha=float(
            self.newton_hessian_regularizer), beta=1.0, masked=True)

        return newton_hessian

    def get_object(self, obj_idx):
        r"""Get a particular object in the scene by its id.

        Args:
            obj_idx (int): Id of object

        Returns:
            SimulatedObject: Simulated Object used by the scene. Also contains ref to simplicits object.
        """
        return self.sim_obj_dict[obj_idx]

    def get_object_deformed_pts(self, obj_idx, points=None):
        r"""Applies linear blend skinning using object's transformation to points provided. By default, points = sim_object.pts

        Args:
            obj_idx (int): Id of object being transformed
            points (torch.Tensor, optional): Points on the object to be transformed.
                                            Defaults to None, which means using *all* object's points.

        Returns:
            torch.Tensor: Transformed points
        """
        obj = self.get_object(obj_idx)
        if points == None:
            points = obj.sample_pts
            
        tfms = self.get_object_transforms(obj_idx)[:, :3, :] # make 4x4 to 3x4

        return weight_function_lbs(points, tfms.unsqueeze(0), obj.simplicits_object.skinning_weight_function).squeeze()

    def get_object_deformation_gradient(self, obj_idx, points=None):
        r"""Applies linear blend skinning using object's transformation to points provided. By default, points = sim_object.pts

        Args:
            obj_idx (int): Id of object being transformed
            points (torch.Tensor, optional): Points on the object to be transformed.
                                            Defaults to None, which means using *all* object's points.

        Returns:
            torch.Tensor: Transformed points
        """
        obj = self.get_object(obj_idx)
        if points == None:
            points = obj.sample_pts

        tfms = self.get_object_transforms(
            obj_idx)[:, :3, :]  # make 4x4 to 3x4

        fcn_get_x = partial(weight_function_lbs, tfms=tfms.unsqueeze(0),
                            fcn=obj.simplicits_object.skinning_weight_function)
        
        Fs = kaolin.physics.utils.finite_diff_jac(fcn_get_x, points, eps=1e-7)

        return Fs

    def _get_scene_ready_for_forces(self):
        r"""Prepares the scene for simulation. Updates any forces that have changed, or objects that have been added/removed.
        """
        self._compute_sim_constants()
        self._create_sim_variables()
        self._ready_for_forces = True
    
    def run_sim_step(self):
        r"""Runs a single simulation step.
        """
        if not self._ready_for_forces:
            raise RuntimeError("Forces need to be set")
        #### Todo: Detect Collisions here? Or Maybe in NM like Gilles ? ####
        self.detect_collision(self.sim_z)
        ###########################################################

        self.sim_z_prev = wp.clone(self.sim_z)

        more_partial_newton_E = partial(
            self._newton_E, wp_B=self.sim_B, wp_BMB=self.sim_BMB, dt=self.timestep, wp_dFdz=self.sim_dFdz,
            defo_grad_energies=None, pt_wise_energies=None, wp_z_prev=self.sim_z_prev, wp_z_dot=self.sim_z_dot)
        more_partial_newton_G = partial(
            self._newton_G, wp_B=self.sim_B, wp_BMB=self.sim_BMB, dt=self.timestep, wp_dFdz=self.sim_dFdz,
            defo_grad_gradients=None, pt_wise_gradients=None, wp_z_prev=self.sim_z_prev, wp_z_dot=self.sim_z_dot)
        more_partial_newton_H = partial(
            self._newton_H, wp_B=self.sim_B, wp_BMB=self.sim_BMB, dt=self.timestep, wp_dFdz=self.sim_dFdz,
            defo_grad_hessians=None, pt_wise_hessians=None, wp_z_prev=self.sim_z_prev, wp_z_dot=self.sim_z_dot)

        ###########################################################

        self.sim_z = newtons_method(
            self.sim_z,
            more_partial_newton_E,
            more_partial_newton_G,
            more_partial_newton_H,
            bounds_fcn=self.compute_collision_bounds,
            preconditioner_fcn= None, # TODO: Fix this cholesky preconditioner, sometimes you get Nans-> self.build_preconditioner,
            Pt=self.sim_Pt,
            P=self.sim_P,
            nm_max_iters=self.max_newton_steps,
            cg_tol=self.cg_tol,
            cg_iters=self.cg_iters,
            conv_tol=self.conv_tol,
            direct_solve=self.direct_solve)

        self.sim_z_dot = wp.from_torch(
            (wp.to_torch(self.sim_z) - wp.to_torch(self.sim_z_prev)) / self.timestep)


        self.current_sim_step += 1
