# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Literal, Any, Union

import warp as wp
import warp.sparse as wps
import numpy as np
import torch
import kaolin

from functools import partial

from ..utils import warp_utilities, torch_utilities

from ..common import Collision, Gravity, Floor, Boundary
from ..materials import NeohookeanElasticMaterial
from ..materials.material_utils import get_defo_grad, to_lame
from ..common.optimization import newtons_method
from .precomputed import sparse_lbs_matrix, sparse_dFdz_matrix
from .skinning import standard_lbs
from .training import SkinnedPointsProtocol, SkinnedPhysicsPoints, SimplicitsObject


logger = logging.getLogger(__name__)

__all__ = [
    'SimulatedObject',
    'SimplicitsScene',
]

class SimulatedObject(SkinnedPhysicsPoints):
    """
    Class containing minimal information for a simulatable simplicits object AND ALSO
    all the variables and state variables needed to run actual simulation.

    This class contains time-varying simulation state and is internal to the
    Simulator logic.

    """
    def __init__(self, pts, yms, prs, rhos, appx_vol, skinning_weights, dwdx,
                 renderable: SkinnedPointsProtocol = None,
                 init_transform=None,
                 is_kinematic=False):
        super().__init__(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol,
                         skinning_weights=skinning_weights, dwdx=dwdx,
                         renderable=renderable)
        self.init_transform = init_transform
        self.is_kinematic = is_kinematic

        # TODO(vismay): need guidance on qp vs. cp here (seems CP should be in scene, but are different from SimulatedObject,
        # which is mostly concerned with pt energies; maybe SimulatedObject could optionally take cp_pts in addition,
        # or collision should be treated separately (e.g. what if we don't use a pt-based method?)
        self.num_qp = self.pts.shape[0]
        self.num_cp = self.pts.shape[0]

        # TODO(vismay): should we allow non-uniform vols on construction instead?
        self.sample_vols = torch.full(
            (self.num_qp,), float(self.appx_vol / self.num_qp),
            device=self.device, dtype=self.dtype)  # uniform integration sampling over vol
        self.sample_masses = (self.appx_vol / self.num_qp) * self.rhos

        self.B = sparse_lbs_matrix(
            wp.from_torch(self.skinning_weights),
            wp.from_torch(self.pts, dtype=wp.vec3))
        self.B.nnz_sync()

        if is_kinematic:
            num_rows = 9 * self.pts.shape[0]
            num_cols = 12 * self.skinning_weights.shape[1]
            self.dFdz = wps.bsr_zeros(num_rows, num_cols, wp.float32)
        else:
            self.dFdz = sparse_dFdz_matrix(
                self.skinning_weights.detach().cpu().numpy(),
                self.dwdx.detach().cpu().numpy(),
                self.pts.detach().cpu().numpy())
        self.dFdz.nnz_sync()

        self._B_dense = None
        self._dFdz_dense = None

        # Let's initialize simulation state variables
        self.z = None
        self.z_prev = None
        self.z_dot = None
        self.reset_sim_state()


    @property
    def B_dense(self):
        if self._B_dense is None:
            self._B_dense = warp_utilities._bsr_to_torch(self.B).to_dense()
        return self._B_dense

    @property
    def dFdz_dense(self):
        if self._dFdz_dense is None:
            if self.is_kinematic:
                self._dFdz_dense = torch.zeros(self.dFdz.shape, device=self.device, dtype=self.dtype)
            else:
                self._dFdz_dense = warp_utilities._bsr_to_torch(self.dFdz).to_dense()
        return self._dFdz_dense

    @classmethod
    def from_skinned_physics_points(cls, phys_pts: SkinnedPhysicsPoints, init_transform, is_kinematic=False):
        """
        Creates simulation object given skinned physics points, minimal data needed for Simplicits simulation
        (e.g. this data can be read from disk).

        Args:
            phys_pts (SkinnedPhysicsPoints): SkinnedPhysicsPoints object to use to define SimulatedObject properties.
            init_transform: Initial transform of the SimulatedObject.
            is_kinematic: If true, the object will be kinematic in the scene. Default: False.

        Returns:
            (SimulatedObject): SimulatedObject object.
        """
        return cls(pts=phys_pts.pts, yms=phys_pts.yms, prs=phys_pts.prs,
                   rhos=phys_pts.rhos, appx_vol=phys_pts.appx_vol,
                   skinning_weights=phys_pts.skinning_weights, dwdx=phys_pts.dwdx,
                   renderable=phys_pts.renderable,
                   init_transform=init_transform, is_kinematic=is_kinematic)

    def reset_sim_state(self):
        r"""Reset the simulation state. Object's handle transforms are set back to initial deformations.
        This does not reset any material parameters or simplicits object parameters.
        """

        self.z = torch.zeros(self.num_handles * 12, dtype=self.dtype, device=self.device)

        # Sets the final (constant) handle
        if self.init_transform is not None:
            self.z[-12:] = self.init_transform.flatten()

        self.z_prev = self.z.clone().detach()
        self.z_dot = torch.zeros_like(self.z, device=self.device)

    def __str__(self):
        r"""String describing object.

        Returns:
            string: String description of object
        """
        super_str = super().__str__()[:-1]
        return f"{super_str}, kinematic={self.is_kinematic}, init_transform={self.init_transform})"

class SimplicitsScene:
    def __init__(self, device='cuda', 
        dtype=torch.float, 
        direct_solve=True, # TODO(vismay): this arg is not used
        use_cuda_graphs=False,  # TODO(vismay): this arg is not used
        timestep=0.03, 
        max_newton_steps=5, 
        max_ls_steps=10, 
        newton_hessian_regularizer=1e-4,  # TODO(vismay): this arg is not used
        cg_tol=1e-4,                      # TODO(vismay): this arg is not used
        cg_iters=100,                     # TODO(vismay): this arg is not used
        conv_tol=1e-4):                   # TODO(vismay): this arg is not used
        r"""Initializes a simplicits scene. SimplicitsObjects can be added to the scene. Scene forces such as floor and gravity can be set on the scene

        Args:
            device (str, optional): Defaults to 'cuda'.
            dtype (torch.dtype, optional): Defaults to torch.float.
            direct_solve (bool, optional): Whether to use direct solve for linear system. Defaults to True.
            use_cuda_graphs (bool, optional): Whether to use cuda graphs. Defaults to False.
            timestep (float, optional): Sim time-step. Defaults to 0.03.
            max_newton_steps (int, optional): Newton steps used in time integrator. Defaults to 5.
            max_ls_steps (int, optional): Line search steps used in time integrator. Defaults to 10.
            newton_hessian_regularizer (float, optional): Regularizer for hessian. Defaults to 1e-4.
            cg_tol (float, optional): Tolerance for conjugate gradient. Defaults to 1e-4.
            cg_iters (int, optional): Maximum number of conjugate gradient iterations. Defaults to 100.
            conv_tol (float, optional): Newtons Method convergence tolerance. Defaults to 1e-4.
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

        self.current_id = 0
        self.sim_obj_dict = {}  # id: SimulatedObject

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

    def _compute_sim_constants(self):  # pragma: no cover
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
        for object_id, object in self.sim_obj_dict.items():
            _num_qp.append(object.num_qp)
            # TODO: here, need guidance on how collision points will/should be treated
            _num_cp.append(object.num_cp)
            _num_handles.append(object.num_handles)

            # Mapping from object to z and x indices
            _object_to_z_map[object_id] = wp.array(
                np.arange(z_index, z_index + object.num_handles * 12), dtype=wp.int32)
            _object_to_qp_map[object_id] = wp.array(
                np.arange(x_index, x_index + object.num_qp), dtype=wp.int32)
            # Add torch array full of object id to list
            _qp_to_object_map.append(torch.full(
                (object.num_qp,), object_id, dtype=torch.int32, device=self.device))
            _z_to_object_map.append(torch.full(
                (object.num_handles * 12,), object_id, dtype=torch.int32))

            if object.is_kinematic:
                _kin_obj_list.append(object_id)
                _kin_obj_to_z_map[object_id] = wp.array(
                    np.arange(z_index, z_index + object.num_handles * 12), dtype=wp.int32)
                _kin_obj_to_qp_map[object_id] = wp.array(
                    np.arange(x_index, x_index + object.num_qp), dtype=wp.int32)

            z_index += object.num_handles * 12
            x_index += object.num_qp

            # Note: our object is already sampled; Scene does not need to keep **all** the pts.
            _stacked_pts.append(object.pts)
            _stacked_rhos.append(object.rhos)
            _stacked_vols.append(object.sample_vols)
            _stacked_masses.append(object.sample_masses)
            _stacked_yms.append(object.yms)
            _stacked_prs.append(object.prs)
            _stacked_sparse_B.append(object.B)
            _stacked_sparse_dFdz.append(object.dFdz)
            _stacked_skinning_weights.append(object.skinning_weights)
        
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
            temp_sim_Pt = warp_utilities._warp_csr_from_torch_dense(torch_utilities.create_projection_matrix(
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
        temp_B = warp_utilities._block_diagonalize(_stacked_sparse_B)
        self.sim_B = wps.bsr_copy(temp_B, block_shape=(1, self._dof_bs))

        self.sim_BMB = wps.bsr_transposed(self.sim_B)@self.sim_M@self.sim_B

        # Scene dFdz Matrix
        temp_dFdz = warp_utilities._block_diagonalize(
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

    def _create_sim_variables(self):  # pragma: no cover
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

        Note:
           This method reset the scene

        Args:
            object_id (int): Id of the object to set the initial transform of
            init_transform (torch.Tensor):
                4x4 torch tensor specifying object's initial skinning transform. 
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
        Returns the current 4x4 padded *relative* transforms of the object's handles. Add identity to the result get standard transform.
        
        Args:
            object_id (int): Id of the object to get the transforms of

        Returns:
            torch.Tensor: Torch tensor of size :math:`(\text{num_handles}, 4, 4)` for relative transforms.
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

    def _add_object(self, simulated_object: SimulatedObject):
        if self._ready_for_forces:
            raise RuntimeError("Cannot add object after a force is set, please create a complete scene with all objects first")

        self.sim_obj_dict[self.current_id] = simulated_object
        self.current_id += 1
        return self.current_id - 1

    def add_object(self, sim_object: Union[SimplicitsObject, SkinnedPhysicsPoints], num_qp=None, init_transform=None, is_kinematic=False, renderable_pts=None):
        r"""Adds a simplicits object to the scene as a SimulatedObject. Can add a just trained SimplicitsObject which
        contains a skinning weight field, or can also accept a baked version, sufficient for simulation.

        Optionally allows the scene to keep track of a rendered entity, such as all the Gaussian points.

        Args:
            sim_object (SimplicitsObject | SkinnedPhysicsPoints): trained simplicits object or already sampled skinned points, e.g. from USD file
            num_qp (int, optional): Number of quadrature points (sample points to integrate over). If not provided, the object will not be subsampled.
            init_transform (torch.Tensor): 3x4 or 4x4 torch tensor specifying object's initial skinning transform.
                                            This argument takes a standard transformation, not a delta.
                                            Subsequently, the Identity matrix is subtracted from it and the delta transform is saved.
            is_kinematic (bool): Object is kinematic if it is not solved for during dynamics simulation.
            renderable_pts (torch.Tensor, optional): Points for rendering (e.g. Gaussian splat positions, in :math:`m`).
                When provided and sim_object is a SimplicitsObject, skinning weights are baked for these
                points and stored in the SimulatedObject for use with get_object_deformed_pts(..., points='rendered').
                This is not to be used with already baked SkinnedPhysicsPointsProtocol.
        """
        # Check if init transform is a 3x4 or 4x4 tensor, convert to 3x4 if necessary
        # Subtract identity transform from init transform to get relative transform
        if torch.is_tensor(init_transform):
            relative_transform = torch_utilities.standard_transform_to_relative(init_transform)
        else:
            relative_transform = torch.zeros(3, 4, device=self.device, dtype=self.dtype)

        if isinstance(sim_object, SimplicitsObject):
            # For SimplicitsObject, bake directly with num_qp to avoid double subsampling
            assert not num_qp is None, "'num_qp' must be provided with SimplicitsObject"
            baked = sim_object.bake(num_qps=num_qp, renderable_pts=renderable_pts)
            simulated_object = SimulatedObject.from_skinned_physics_points(
                baked, init_transform=relative_transform, is_kinematic=is_kinematic)
        else:
            # For already baked SkinnedPhysicsPointsProtocol, subsample if needed
            assert renderable_pts is None, "'renderable_pts' are not supported for already baked SkinnedPhysicsPointsProtocol"
            sampled = sim_object.subsample(num_pts=num_qp) if num_qp is not None else sim_object
            simulated_object = SimulatedObject.from_skinned_physics_points(
                sampled, init_transform=relative_transform, is_kinematic=is_kinematic)

        return self._add_object(simulated_object)
    
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
        r"""Sets the gravity in the scene. Applies it to all objects in scene.

        Args:
            acc_gravity (torch.Tensor, optional): Gravity acceleration. Defaults to torch.tensor([0, 9.8, 0]) with acceleration due to gravity in the downward y direction.
            gravity_coeff (float, optional): Gravity coefficient. Defaults to 1.0.
        """
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
            flip_floor (bool, optional): Flips the direction of the floor. Defaults to False.
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

    def set_object_boundary_condition(self, obj_idx, name, fcn, bdry_penalty=10000.0, pinned_x=None):
        r"""Sets boundary condition for object in scene

        Args:
            obj_idx (int): Id of object
            name (str): Boundary condition name
            fcn (Callable): Function that defines which indices the boundary condition applies to. fcn should return a boolean :math:`(n)` vector where bdry indices are 1.
            bdry_penalty (float): Boundary condition penalty coefficient.
            pinned_x (torch.Tensor, optional): Pinned positions of the boundary condition. Used for setting the boundary to a specific position. Default: The pinned positions are set from the current object's positions.
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
                          collision_penalty=1000.0, 
                          max_contact_pairs=10000,
                          friction=0.5):
        r"""Sets collision for object in scene

        Args:
            collision_particle_radius (float, optional): Scene-wide collision particle radius at which penalty begins to apply. Change this depending on the size of the object. Defaults: 0.1.
            detection_ratio (float, optional): Collision detection radius described as a ratio relative to the collision_particle_radius. Defaults: 1.5 times the collision_particle_radius.
            impenetrable_barrier_ratio (float, optional): Collision barrier radius described as a ratio relative to the collision_particle_radius. Defaults: 0.25 times the collision_particle_radius.
            collision_penalty (float, optional): Controls the stiffness of the collision interaction. Defaults: 1000 times the collision_particle_radius.
            max_contact_pairs (int, optional): Maximum number of contact pairs to detect. If this is too low, some contacts may be missed. If this is too high, memory may run out/jacobian may be too large. Defaults: 10000 contact pairs.
            friction (float, optional): Friction coefficient. Defaults: 0.5.
        """

        if not self._ready_for_forces:
            self._get_scene_ready_for_forces()

        collision_struct = Collision(
            dt=self.timestep,
            collision_particle_radius=collision_particle_radius,
            detection_ratio=detection_ratio,
            impenetrable_barrier_ratio=impenetrable_barrier_ratio,
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

        self._detect_collision(self.sim_z)

    def _detect_collision(self, z):  # pragma: no cover
        r"""Resets the collision jacobian when new contact pairs are found.
        
        Args:
            z (torch.Tensor): Current scene state.
        
        Note:
            This method is called internally by the scene to detect collisions.
            It is not meant to be called by the user.
        """
        
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
        collision_struct.get_jacobian(
            cp_w=self.sim_skinning_weights, cp_x0=self.sim_pts, cp_is_static=self.qp_is_kinematic)

    def _build_preconditioner(self, lhs):  # pragma: no cover
        return warp_utilities._build_preconditioner(lhs)

    def _compute_collision_bounds(self, dz, z):  # pragma: no cover
        if "collision" not in self.force_dict or self.force_dict["collision"]["object"] is None:
            return None

        assert dz.shape[0] == self.sim_B.shape[1]

        collision_struct = self.force_dict["collision"]["object"]
        if collision_struct.num_contacts == 0:
            return None

        dx = wp.array((self.sim_B@z), dtype=wp.vec3)

        delta_dx = wp.array((self.sim_B@dz), dtype=wp.vec3)

        wp_bounds = collision_struct.get_bounds(cp_delta_dx=delta_dx,  # B*z_k, z_k = z at nm step 0
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

    def _assemble_energies(self, z, delta_dz):  # pragma: no cover
        x0 = self.sim_pts

        # copy to fixed memory location so we can use in graph
        wp.copy(src=z, dest=self._eval_z)
        wp.copy(src=delta_dz, dest=self._eval_delta_dz)

        def eval_fixed_energies():
            # Names of gradients to assemble
            pt_names = list(self.force_dict["pt_wise"].keys())
            defo_grad_names = list(self.force_dict["defo_grad_wise"].keys())

            # Get grad function input variables
            F_ele = get_defo_grad(
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

    def _assemble_gradients(self, z):  # pragma: no cover
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
            F_ele = get_defo_grad(
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

    def _assemble_hessians(self, z):  # pragma: no cover
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
        F_ele = get_defo_grad(
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
        for obj_id, obj in self.sim_obj_dict.items():
            qp_i = wp.to_torch(self.object_to_qp_map[obj_id])

            H_ii = torch_utilities.hess_reduction(obj.B_dense, scene_d2Edx2[qp_i, :, :]) + torch_utilities.hess_reduction(
                obj.dFdz_dense, scene_d2EdF2[qp_i, :, :])

            hess_list.append((obj_id, obj_id, H_ii))

        # this is a sparse matrix
        # H = warp_utilities._assemble_global_hessian(
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
            # cH = warp_utilities._wp_hessian_reduction(cJt, collision_hess, cJ)
            # cH33 = wps.bsr_copy(cH, block_shape=(3, 3))
            # wps.bsr_axpy(cH33, H, alpha=1.0, beta=1.0)

        # this is a sparse matrix
        H = warp_utilities._assemble_global_hessian(
            hess_list, self.object_to_z_map, z, block_size=self._dof_bs)

        return H

    @staticmethod
    def _displacement_delta(wp_z, wp_z_prev, wp_z_dot, dt):  # pragma: no cover
        r"""Timestep displacement update, to use in inertia computations

        Args:
            wp_z (wp.array): Transforms
            wp_z_prev (wp.array): Previous transforms
            wp_z_dot (wp.array): Time derivative of transforms
            dt (float): Timestep
        """

        delta_dz = wp.empty_like(wp_z)
        wp.launch(warp_utilities._displacement_delta_kernel, dim=delta_dz.shape, inputs=[
                  dt, wp_z, wp_z_prev, wp_z_dot], outputs=[delta_dz])
        return delta_dz

    def _newton_E(self, wp_z, wp_z_prev, wp_z_dot,
                  wp_B, wp_BMB, dt, wp_dFdz, defo_grad_energies=None, pt_wise_energies=None):  # pragma: no cover
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

    def _newton_G(self, wp_z, wp_z_prev, wp_z_dot, wp_B, wp_BMB, dt, wp_dFdz, defo_grad_gradients=None, pt_wise_gradients=None):  # pragma: no cover
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

    def _newton_H(self, wp_z, wp_z_prev, wp_z_dot, wp_B, wp_BMB, dt, wp_dFdz, defo_grad_hessians=None, pt_wise_hessians=None):  # pragma: no cover
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

    def get_object_deformed_pts(self, obj_idx, points: Literal['rendered', 'simulated'] = 'simulated'):
        r"""Applies linear blend skinning using object's transformation to points provided.

        Args:
            obj_idx (int): Id of object being transformed
            points (str, optional): Which point set to transform. One of 'simulated' (default)
                or 'rendered'. For 'rendered', the object must have been added with renderable_pts.

        Returns:
            torch.Tensor: Transformed points
        """
        sim_obj = self.get_object(obj_idx)

        if points == 'rendered':
            if sim_obj.renderable is None:
                raise ValueError(
                    f'Object {obj_idx} has no renderable points. '
                    f'Pass renderable_pts when calling add_object().')
            pts = sim_obj.renderable.pts
            skinning_weights = sim_obj.renderable.skinning_weights
        elif points == 'simulated':
            pts = sim_obj.pts
            skinning_weights = sim_obj.skinning_weights
        else:
            raise ValueError('Only supports "rendered" or "simulated" points')

        tfms = self.get_object_transforms(obj_idx)[:, :3, :]  # make 4x4 to 3x4
        return standard_lbs(pts, tfms.unsqueeze(0), skinning_weights).squeeze()

    def get_object_point_transforms(self, obj_idx,
                                    points: Literal['rendered', 'simulated'] = 'simulated'):
        r"""
        Returns the absolute transform of the points of an object.

        Args:
            obj_idx (int): Id of the object.
            points (str, optional): Which point set to query. One of 'simulated' (default) or
                'rendered'. For 'rendered', the object must have been added with renderable_pts.

        Returns:
            torch.Tensor: Torch tensor of size :math:`(\text{num_points}, 4, 4)` for transforms.
        """
        sim_obj = self.get_object(obj_idx)

        if points == 'rendered':
            if sim_obj.renderable is None:
                raise ValueError(
                    f'Object {obj_idx} has no renderable skinning weights. '
                    f'Pass renderable skinning weights when calling add_object().')
            skinning_weights = sim_obj.renderable.skinning_weights
        elif points == 'simulated':
            skinning_weights = sim_obj.skinning_weights
        else:
            raise ValueError('Only supports "rendered" or "simulated" points')

        transforms = self.get_object_transforms(obj_idx)

        # N x 4 x 4 = sum((N x H x 1 x 1) * (1 x H x 4 x 4), dim=1)
        per_pt_transforms = torch.sum(skinning_weights.unsqueeze(-1).unsqueeze(-1) * transforms, dim=1)

        per_pt_transforms[:, :3, :3] += torch.eye(3, 3, dtype=per_pt_transforms.dtype, device=per_pt_transforms.device).unsqueeze(0)

        return per_pt_transforms

    def _get_scene_ready_for_forces(self):  # pragma: no cover
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
        self._detect_collision(self.sim_z)
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
            bounds_fcn=self._compute_collision_bounds,
            preconditioner_fcn= None, # TODO: Fix this cholesky preconditioner, sometimes you get Nans-> self._build_preconditioner,
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
