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

from kaolin.physics.simplicits.network import SimplicitsMLP
import kaolin.physics.simplicits.utils as simplicits_utils
import kaolin.physics.simplicits.precomputed as precomputed
import kaolin.physics.utils as phys_utils
import kaolin.physics.materials as materials
import kaolin.physics.utils.optimization as optimization
from kaolin.physics.simplicits.simplicits_scene_forces import *
from kaolin.physics.simplicits.losses_warp import compute_losses_warp
from kaolin.physics.simplicits.losses import compute_losses
from kaolin.physics.simplicits.losses_warp import compute_losses_warp
from kaolin.physics.simplicits.losses import compute_losses
import os
import torch
import numpy as np
import logging
from functools import partial


logger = logging.getLogger(__name__)


__all__ = [
    'SimplicitsObject',
    'SimplicitsScene',
]


class SimplicitsObject:
    def __init__(self, pts, yms, prs, rhos, appx_vol, num_handles=10, num_samples=1000, model_layers=6,
                 training_batch_size=10, normalize_for_training=True, warp_training=False):
        r""" Easy to use wrapper for initializing and training a simplicits object based on the paper https://research.nvidia.com/labs/toronto-ai/simplicits/

        Args:
            pts (torch.Tensor): Object's points (ideally over the volume), sampled and used as integration primitives, of shape :math:`(n, 3)`
            yms (torch.Tensor): Youngs modulus. Either pass in a :math:`(n)` tensor, or one value which will be broadcasted.
            prs (torch.Tensor): Poisson ratio. Either pass in a :math:`(n)` tensor, or one value  which will be broadcasted.
            rhos (torch.Tensor): Density. Either pass in a :math:`(n)` tensor, or one value  which will be broadcasted.
            appx_vol (torch.Tensor): Approximated volume of the object, of shape 0d tensor.
            num_handles (int, optional): Number of skinning handles (reduced DOFs) over the object. Defaults to 10.
            num_samples (int, optional): Training parameter. Number of sample points (integration points) over the object. Defaults to 1000.
            model_layers (int, optional): Training parameter. Layers in the simplicits network. Defaults to 6. 
                                            Too few causes poor deformations. Too many slows computation.
            learning_rate (torch.Tensor, optional): Training parameter. Network's learning rate. Defaults to 1e-3.
            training_batch_size (int, optional): Training parameter. Defaults to 10.
        """
        # TODO: allow to specify just floats, and this will create tensors

        self.default_device = pts.device
        self.default_dtype = pts.dtype

        self.num_handles = num_handles
        self.pts = torch.as_tensor(pts, device=self.default_device, dtype=self.default_dtype)
        self.yms = torch.as_tensor(yms, device=self.default_device, dtype=self.default_dtype).reshape(-1, 1)
        self.prs = torch.as_tensor(prs, device=self.default_device, dtype=self.default_dtype).reshape(-1, 1)
        self.rhos = torch.as_tensor(rhos, device=self.default_device, dtype=self.default_dtype).reshape(-1, 1)
        self.appx_vol = torch.as_tensor(appx_vol, device=self.default_device, dtype=self.default_dtype)

        self.bb_max = torch.max(self.pts, dim=0).values
        self.bb_min = torch.min(self.pts, dim=0).values

        # Normalize the appx vol of object
        norm_bb_max = torch.max((self.pts - self.bb_min) / (self.bb_max - self.bb_min),
                                dim=0).values  # get the bb_max of the normalized pts
        norm_bb_min = torch.min((self.pts - self.bb_min) / (self.bb_max - self.bb_min),
                                dim=0).values  # get the bb_min of the normalized pts

        bb_vol = (self.bb_max[0] - self.bb_min[0]) * (self.bb_max[1] -
                                                      self.bb_min[1]) * (self.bb_max[2] - self.bb_min[2])
        norm_bb_vol = (norm_bb_max[0] - norm_bb_min[0]) * (norm_bb_max[1] -
                                                           norm_bb_min[1]) * (norm_bb_max[2] - norm_bb_min[2])

        norm_appx_vol = self.appx_vol * (norm_bb_vol / bb_vol)

        # If normalize_for_training, then
        # Re-approximate the vol to fit the normalized object
        self.normalize_for_training = normalize_for_training
        self.normalized_pts = self.pts
        if (self.normalize_for_training):
            norm_appx_vol = self.appx_vol * (norm_bb_vol / bb_vol)
        else:
            norm_appx_vol = self.appx_vol

        self.num_samples = num_samples
        self.training_layers = model_layers
        if warp_training:
            self.compute_losses = partial(compute_losses_warp,
                                          batch_size=training_batch_size,  # TODO: maybe pass into train() below?
                                          num_handles=self.num_handles,
                                          appx_vol=norm_appx_vol,
                                          num_samples=self.num_samples)
        else:
            self.compute_losses = partial(compute_losses,
                                          batch_size=training_batch_size,  # TODO: maybe pass into train() below?
                                          num_handles=self.num_handles,
                                          appx_vol=norm_appx_vol,
                                          num_samples=self.num_samples)

        self.model = None
        if self.num_handles == 0:
            self.model_plus_rigid = lambda pts: torch.ones((pts.shape[0], 1), device=self.default_device)
        else:
            if (self.normalize_for_training):
                self.model_plus_rigid = lambda pts: torch.cat((self.model(
                    (pts - self.bb_min) / (self.bb_max - self.bb_min)), torch.ones((pts.shape[0], 1), device=self.default_device)), dim=1)
            else:
                self.model_plus_rigid = lambda pts: torch.cat(
                    (self.model(pts), torch.ones((pts.shape[0], 1), device=self.default_device)), dim=1)

    def save_model(self, pth):
        r"""Saves the Simplicits network model (not including rigid mode)

        Args:
            pth (str): Path to simplicits model file
        """
        torch.save(self.model, pth)

    def load_model(self, pth):
        r"""Loads the Simplicits network model from file. Adds rigid mode.

        Args:
            pth (str): Path to simplicits model file
        """
        self.model = torch.load(pth).to(device=self.default_device)
        if self.num_handles == 0:
            self.model_plus_rigid = lambda pts: torch.ones((pts.shape[0], 1), device=self.default_device)
        else:
            if (self.normalize_for_training):
                self.model_plus_rigid = lambda pts: torch.cat((self.model(
                    (pts - self.bb_min) / (self.bb_max - self.bb_min)), torch.ones((pts.shape[0], 1), device=self.default_device)), dim=1)
            else:
                self.model_plus_rigid = lambda pts: torch.cat(
                    (self.model(pts), torch.ones((pts.shape[0], 1), device=self.default_device)), dim=1)

    def train(self, num_steps=10000, lr_start=1e-3, lr_end=1e-3, le_coeff=1e-1, lo_coeff=1e6, log_every=1000):
        r"""Trains object. If object has already been trained, calling this function will replace the previously trained results.

        Args:
            num_steps (int, optional): Number of training steps. Defaults to 10000.
            lr_start (float, optional): Learning rate at start of training. Defaults to 1e-3.
            lr_end (float, optional): Learning rate at end of training. Defaults to 1e-3.
            le_coeff (float, optional): Training parameter. Elasticity loss coefficient. Defaults to 1e-1.
            lo_coeff (float, optional): Training parameter. Orthogonality loss coefficient. Defaults to 1e6.
            log_every (int, optional): Number of steps after which to log the status of the training

        Returns:
            list, optional: List of pair-wise loss float values at each `log_every` number of steps (defaults to 1000)

        """
        if self.num_handles == 0:
            # Rigid object, no training
            return

        # create a new model at training time
        self.model = SimplicitsMLP(3, 64, self.num_handles, self.training_layers)
        self.model.to(self.default_device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr_start)

        self.model.train()
        e_logs = []
        for i in range(num_steps):
            optimizer.zero_grad()
            le, lo = self.compute_losses(self.model, self.normalized_pts, self.yms, self.prs,
                                         self.rhos, float(i / num_steps), le_coeff=le_coeff, lo_coeff=lo_coeff)
            loss = le + lo

            loss.backward()
            optimizer.step()

            # Update learning rate. Linearly interpolate from lr_start to lr_end
            for grp in optimizer.param_groups:
                grp['lr'] = lr_start + float(i / num_steps) * (lr_end - lr_start)

            if i % log_every == 0:
                logger.info(f'Training step: {i}, le: {le.item()}, lo: {lo.item()}')
                e_logs.append((le.item(), lo.item()))

        self.model.eval()
        return e_logs


# private utility class
class SimulatedObject:
    def __init__(self, obj: SimplicitsObject, num_cub_pts=1000, init_tfm=None):
        r""" Initialize a simulated object from a simplicits object. Users should avoid direct access to this class. 
            Instead use the SimulatedObject's obj_idx in the getters/setters for its parameters, forces and materials in the Scene to update/modify this object.

            Args:
                obj (SimplicitsObject): Saves the Simplicits Object to be used in the sim by reference here
                num_cub_pts (int, optional): Number of cubature points (integration primitives). Defaults to 1000
        """
        self.simplicits_object = obj
        self.num_cub_pts = num_cub_pts
        self._sample_cubatures(num_cub_pts=num_cub_pts)
        self.reset_sim_state(init_tfm)
        self._set_sim_constants()

        r""" Structure the force_dict like this.
        force_dict = {
                "pt_wise":{
                    "floor":{
                        "energy": floor_energy_fcn,
                        "gradient": floor_gradient_fcn,
                        "hessian": floor_hessian_fcn,
                    },
                    "gravity":{
                        "energy": gravity_energy_fcn,
                        "gradient": gravity_gradient_fcn,
                        "hessian": gravity_hessian_fcn,
                    },
                    "other":{
                        "energy": other_energy_fcn,
                        "gradient": other_gradient_fcn,
                        "hessian": other_hessian_fcn,
                    },
                },
                "defo_grad_wise":{
                    "material":{
                        "energy": material_energy_fcn,
                        "gradient": material_gradient_fcn,
                        "hessian": material_hessian_fcn,
                    },
                    "muscle":{
                        "energy": muscle_energy_fcn,
                        "gradient": muscle_gradient_fcn,
                        "hessian": muscle_hessian_fcn,
                    },
                    "other":{
                        "energy": other_energy_fcn,
                        "gradient": other_gradient_fcn,
                        "hessian": other_hessian_fcn,
                    }
                }

            }
        """
        self.integration_sampling = (self.simplicits_object.appx_vol /
                                     self.sim_pts.shape[0]).to(dtype=self.sim_pts.dtype)  # uniform integration sampling over vol
        self.force_dict = {"pt_wise": {},
                           "defo_grad_wise": {}}

        self.set_materials(self.sim_yms, self.sim_prs)

    def __str__(self):
        r"""String describing object.

        Returns:
            string: String description of object
        """
        unique_densities = torch.unique(self.sim_rhos).tolist()
        unique_yms = torch.unique(self.sim_yms).tolist()
        unique_prs = torch.unique(self.sim_prs).tolist()
        appx_vol = self.simplicits_object.appx_vol.item()
        return f"SimulatedObject(Densities={unique_densities}, Yms={unique_yms}, Prs={unique_prs}, AppxVol={appx_vol}, num_cub_pts={self.sim_pts.shape[0]})"

    def _sample_cubatures(self, num_cub_pts):
        r"""Internal function to sample cubature points over the simplicits object. 
            Currently, we assume uniformal sampling.
            Sets simulation object's material properties.

        Args:
            num_cub_pts (int): Number of cubature points to sample over the mesh.
        """
        self.sample_indices = torch.randint(low=0, high=self.simplicits_object.normalized_pts.shape[0], size=(
            num_cub_pts,), device=self.simplicits_object.default_device)
        self.sim_pts = self.simplicits_object.pts[self.sample_indices]
        self.sim_normalized_pts = self.simplicits_object.normalized_pts[self.sample_indices]
        self.sim_yms = self.simplicits_object.yms[self.sample_indices]
        self.sim_prs = self.simplicits_object.prs[self.sample_indices]
        self.sim_rhos = self.simplicits_object.rhos[self.sample_indices]

    def _set_sim_constants(self):
        # Simulation constants
        r""" Constants req for simulation. Computed once upfront. 
            must be updated when _resample_cubature() is called
        """
        self.x0_flat = self.sim_pts.flatten().unsqueeze(-1).detach()
        self.weights = self.simplicits_object.model_plus_rigid(self.sim_normalized_pts)
        self.M, self.invM = precomputed.lumped_mass_matrix(self.sim_rhos, self.simplicits_object.appx_vol, dim=3)
        self.bigI = torch.tile(torch.eye(3, device=self.simplicits_object.default_device).flatten(
        ).unsqueeze(dim=1), (self.num_cub_pts, 1)).detach()

        self.dFdz = precomputed.jacobian_dF_dz(self.simplicits_object.model_plus_rigid,
                                               self.sim_normalized_pts, self.z).detach()
        self.B = precomputed.lbs_matrix(self.sim_pts, self.weights).detach()
        self.BMB = (self.B.T @ self.M @ self.B).detach()
        self.BinvMB = (self.B.T @ self.invM @ self.B).detach()

    def reset_sim_state(self, init_tfm=None):
        self.z = torch.zeros((self.simplicits_object.num_handles + 1) * 12,
                             dtype=self.simplicits_object.default_dtype, device=self.simplicits_object.default_device).unsqueeze(-1)
        if init_tfm is not None:
            self.z[-12:, 0] = init_tfm.flatten()
        self.z_prev = self.z.clone().detach()
        self.z_dot = torch.zeros_like(self.z, device=self.simplicits_object.default_device)

    def set_boundary_condition(self, name, fcn, bdry_penalty):
        r"""Sets boundary with energy, gradient, hessian for the object.

        Args:
            name (str): Name of the boundary condition used for bookkeeping
            fcn (Callable): Function that defines which indices the boundary condition applies to. fcn should return a boolean :math:`(n)` vector where bdry indices are 1.
            bdry_penalty (float): Boundary condition penalty coefficient.

        Returns:
            kaolin.physics.utils.Boundary: Boundary condition object reference returned for bookkeeping on client-side
        """
        bdry_cond = phys_utils.Boundary()
        bdry_indx = torch.nonzero(fcn(self.sim_pts), as_tuple=False).squeeze()
        bdry_pos = self.sim_pts[bdry_indx, :]
        bdry_cond.set_pinned_verts(bdry_indx, bdry_pos)
        self.force_dict["pt_wise"][name] = {}
        self.force_dict["pt_wise"][name]["force_object"] = bdry_cond

        partial_bdry_e = generate_fcn_simplicits_scene_energy(
            bdry_cond, self.B, coeff=bdry_penalty, integration_sampling=None)
        partial_bdry_g = generate_fcn_simplicits_scene_gradient(
            bdry_cond, self.B, coeff=bdry_penalty, integration_sampling=None)
        partial_bdry_h = generate_fcn_simplicits_scene_hessian(
            bdry_cond, self.B, coeff=bdry_penalty, integration_sampling=None)

        self.force_dict["pt_wise"][name]["energy"] = partial_bdry_e
        self.force_dict["pt_wise"][name]["gradient"] = partial_bdry_g
        self.force_dict["pt_wise"][name]["hessian"] = partial_bdry_h
        return bdry_cond

    def set_materials(self, yms=None, prs=None, rhos=None):
        r"""Sets the material energy, forces, hessian of the simulated object. For now default material is neohookean.

        Args:
            yms (torch.Tensor, optional): Sets Yms of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
            prs (torch.Tensor, optional): Sets Prs of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
            rhos (torch.Tensor, optional): Sets densities of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
        """
        if yms is not None:
            self.sim_yms = yms.reshape(-1, 1)
        if prs is not None:
            self.sim_prs = prs.reshape(-1, 1)

        if rhos is not None:
            self.sim_rhos = rhos.expand(self.sim_rhos.shape[0], 1)
            self._set_sim_constants()

        # Setup Material
        material_object = materials.NeohookeanMaterial(self.sim_yms, self.sim_prs)

        partial_material_e = generate_fcn_simplicits_material_energy(
            material_object, self.dFdz, coeff=1, integration_sampling=self.integration_sampling)
        partial_material_g = generate_fcn_simplicits_material_gradient(
            material_object, self.dFdz, coeff=1, integration_sampling=self.integration_sampling)
        partial_material_h = generate_fcn_simplicits_material_hessian(
            material_object, self.dFdz, coeff=1, integration_sampling=self.integration_sampling)
        self.force_dict["defo_grad_wise"]["material"] = {}
        self.force_dict["defo_grad_wise"]["material"]["energy"] = partial_material_e
        self.force_dict["defo_grad_wise"]["material"]["gradient"] = partial_material_g
        self.force_dict["defo_grad_wise"]["material"]["hessian"] = partial_material_h

    def set_gravity(self, grav):
        r"""Sets gravity energy, forces, hessian for the object.

        Args:
            grav (torch.Tensor): Acceleration due to gravity in [x,y,z] direction as :math:`(x, y, z)` acceleration tensor
        """
        gravity_object = phys_utils.Gravity(rhos=self.sim_rhos, acceleration=grav)

        partial_grav_e = generate_fcn_simplicits_scene_energy(
            gravity_object, self.B, coeff=1, integration_sampling=self.integration_sampling)
        partial_grav_g = generate_fcn_simplicits_scene_gradient(
            gravity_object, self.B, coeff=1, integration_sampling=self.integration_sampling)
        partial_grav_h = generate_fcn_simplicits_scene_hessian(
            gravity_object, self.B, coeff=1, integration_sampling=self.integration_sampling)
        self.force_dict["pt_wise"]["gravity"] = {}
        self.force_dict["pt_wise"]["gravity"]["energy"] = partial_grav_e
        self.force_dict["pt_wise"]["gravity"]["gradient"] = partial_grav_g
        self.force_dict["pt_wise"]["gravity"]["hessian"] = partial_grav_h

    def set_floor(self, floor_height, floor_axis, floor_penalty, flip_floor):
        r"""Sets floor penalty energy, forces, hessian for object.

        Args:
            floor_height (float): Height of the floor, defaults to 0
            floor_axis (int): 0 for x axis, 1 for y axis, 2 for z axis
            floor_penalty (float): Strength of the floor penalty.
        """
        floor_object = phys_utils.Floor(floor_height=floor_height, floor_axis=floor_axis, flip_floor=flip_floor)

        partial_floor_e = generate_fcn_simplicits_scene_energy(
            floor_object, self.B, coeff=floor_penalty, integration_sampling=None)
        partial_floor_g = generate_fcn_simplicits_scene_gradient(
            floor_object, self.B, coeff=floor_penalty, integration_sampling=None)
        partial_floor_h = generate_fcn_simplicits_scene_hessian(
            floor_object, self.B, coeff=floor_penalty, integration_sampling=None)
        self.force_dict["pt_wise"]["floor"] = {}
        self.force_dict["pt_wise"]["floor"]["energy"] = partial_floor_e
        self.force_dict["pt_wise"]["floor"]["gradient"] = partial_floor_g
        self.force_dict["pt_wise"]["floor"]["hessian"] = partial_floor_h

    def remove_force(self, name):
        r"""Remove force from force_dict. Force will no longer apply to object. 

        Args:
            name (string): Name of the force to remove.
        """
        # if force is removed on the scene, remove force on object
        # check if its a pt-wise force
        if name in self.force_dict["pt_wise"]:
            del self.force_dict["pt_wise"][name]
        elif name in self.force_dict["defo_grad_wise"]:
            del self.force_dict["defo_grad_wise"][name]

    def get_all_force_fcns(self, type_wise, level):
        r"""Gets a list of functions from the force_dict.

        Args:
            type_wise (string): Type of force. "pt_wise" or "defo_grad_wise"
            level (string): "energy", "gradient" or "hessian"

        Returns:
            list: list of functions in force_dict of type: `type_wise` and level: `level`
        """
        fcn_list = []
        for name in self.force_dict[type_wise]:
            fcn_list.append(self.force_dict[type_wise][name][level])
        return fcn_list


class SimplicitsScene:

    def __init__(self,
                 device='cuda',
                 dtype=torch.float,
                 timestep=0.03,
                 max_newton_steps=20,
                 max_ls_steps=30):
        r"""Initializes a simplicits scene. This is the entry point to Simplicits easy API. 
        SimplicitsObjects can be added to the scene. 
        Scene forces such as floor and gravity can be set on the scene

        Args:
            device (str, optional): Defaults to 'cuda'.
            dtype (torch.dtype, optional): Defaults to torch.float.
            timestep (float, optional): Sim time-step. Defaults to 0.03.
            max_newton_steps (int, optional): Newton steps used in time integrator. Defaults to 20.
            max_ls_steps (int, optional): Line search steps used in time integrator. Defaults to 30.
        """
        self.default_device = device
        self.default_dtype = dtype

        self.timestep = timestep
        self.current_sim_step = 0

        self.max_newton_steps = max_newton_steps
        self.max_ls_steps = max_ls_steps

        self.current_id = 0
        self.sim_obj_dict = {}

    def add_object(self, sim_object: SimplicitsObject, num_cub_pts=1000, init_tfm=None):
        r"""Adds a simplicits object to the scene as a SimulatedObject.

        Args:
            sim_object (SimplicitsObject): Simplicits object that will be wrapped into a SimulatedObject for this scene.
            num_cub_pts (int, optional): Number of cubature pts (integration primitives) to sample during simulation. Defaults to 1000.
            init_tfm (torch.Tensor, optional): Initial transformation of the object, shape of :math:`[3,4]`. Defaults to None (no initial transformation).

        Returns:
            int: Id of object in the scene.
        """
        self.sim_obj_dict[self.current_id] = SimulatedObject(sim_object, num_cub_pts=num_cub_pts, init_tfm=init_tfm)
        self.current_id += 1
        return self.current_id - 1

    def remove_object(self, obj_idx):
        r"""Removes object from scene

        Args:
            obj_idx (int): Id of object to be removed
        """
        if obj_idx in self.sim_obj_dict:
            del self.sim_obj_dict[obj_idx]

    def get_object(self, obj_idx):
        r"""Get a particular object in the scene by its id.

        Args:
            obj_idx (int): Id of object

        Returns:
            SimulatedObject: Simulated Object used by the scene. Also contains ref to simplicits object.
        """
        return self.sim_obj_dict[obj_idx]

    def reset_object(self, obj_idx, init_tfm=None):
        r"""Resets the state of the object back to default

        Args:
            obj_idx (int): Simulated object Id
            init_tfm (torch.Tensor, optional): Initial transformation of the object, shape of :math:`[3,4]`. Defaults to None.
        """
        if obj_idx in self.sim_obj_dict:
            self.sim_obj_dict[obj_idx].reset_sim_state(init_tfm=init_tfm)

    def remove_object_force(self, obj_idx, name):
        r"""Removes this force for this object.

        Args:
            obj_idx (int): Id of object
            name (str): Force to be removed from the scene
        """
        self.sim_obj_dict[obj_idx].remove_force(name)

    def get_object_boundary_condition(self, obj_idx, name):
        r"""Get boundary condition for object by name.

        Args:
            obj_idx (int): Id of object
            name (str): Boundary condition name

        Returns:
            Boundary: Boundary condition object.
        """
        return self.sim_obj_dict[obj_idx]["pt_wise"][name]["force_object"]

    def set_object_boundary_condition(self, obj_idx, name, fcn, bdry_penalty):
        r"""Sets boundary condition for object in scene

        Args:
            obj_idx (int): Id of object
            fcn (Callable): Function that defines which indices the boundary condition applies to. fcn should return a boolean :math:`(n)` vector where bdry indices are 1.
            name (str): Boundary condition name
            bdry_penalty (float): Boundary condition penalty coefficient.
        """
        self.sim_obj_dict[obj_idx].set_boundary_condition(name, fcn, bdry_penalty)

    def set_object_materials(self, obj_idx, yms=None, prs=None, rhos=None):
        r"""Sets object's material properties

        Args:
            obj_idx (int): Id of object
            yms (torch.Tensor, optional): Sets Yms of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
            prs (torch.Tensor, optional): Sets Prs of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
            rhos (torch.Tensor, optional): Sets densities of simulated object. Pass in :math:`(n)` tensor or one value to get broadcasted. Defaults to None.
        """
        self.sim_obj_dict[obj_idx].set_materials(yms=yms, prs=prs, rhos=rhos)

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
            points = obj.simplicits_object.pts
        return simplicits_utils.weight_function_lbs(points, tfms=obj.z.reshape(-1, 3, 4).unsqueeze(0), fcn=obj.simplicits_object.model_plus_rigid)

    def get_object_deformation_gradient(self, obj_idx, points=None):
        r"""Gets the deformation gradients of the objects integration points

        Args:
            obj_idx (int): Id of object being transformed
            points (torch.Tensor, optional): Calculating the deformation gradients over these pts

        Returns:
            torch.Tensor: Deformation gradients
        """
        obj = self.get_object(obj_idx)
        # F_ele = torch.matmul(obj.dFdz, obj.z) + obj.bigI
        # return F_ele.reshape(-1, 3, 3)
        if points == None:
            points = obj.simplicits_object.pts
        fcn_get_x = partial(simplicits_utils.weight_function_lbs, tfms=obj.z.reshape(-1,
                            3, 4).unsqueeze(0), fcn=obj.simplicits_object.model_plus_rigid)
        Fs = phys_utils.finite_diff_jac(fcn_get_x, points, eps=1e-7)
        return Fs

    def reset(self):
        r"""Resets all objects in the scene, no initial transforms. Use reset_object(obj_idx) for 
            setting objects with initial transforms.
        """
        for idx in self.sim_obj_dict:
            self.sim_obj_dict[idx].reset_sim_state(init_tfm=None)
        return

    def set_scene_floor(self, floor_height=0, floor_axis=1, floor_penalty=10000, flip_floor=False):
        r"""Sets the floor in the scene. Applies it to all objects in scene.

        Args:
            floor_height (float, optional): Floor height. Defaults to 0.
            floor_axis (int, optional): Direction of floor. 0 is x, 1 is y, 2 is z. Defaults to 1.
            floor_penalty (float, optional): Stiffness of floor. Defaults to 10000.
        """
        for idx in self.sim_obj_dict:
            o = self.sim_obj_dict[idx]
            o.set_floor(floor_height, floor_axis, floor_penalty, flip_floor)

    def set_scene_gravity(self, acc_gravity=torch.tensor([0, 9.8, 0])):
        r"""Sets gravity in the scene. Applies it to all objects in scene.

        Args:
            acc_gravity (torch.Tensor, optional): Acceleration due to gravity. Defaults to :math:`\text{torch.tensor([0, 9.8, 0])}`.
        """
        grav = acc_gravity.to(self.default_device)
        for idx in self.sim_obj_dict:
            o = self.sim_obj_dict[idx]
            o.set_gravity(grav)

    def remove_scene_force(self, name):
        r"""Removes this force for all objects in the scene.

        Args:
            name (str): Force to be removed from the scene
        """
        for idx in self.sim_obj_dict:
            o = self.sim_obj_dict[idx]
            o.remove_force(name)

    #################### Backwards Euler Functions###########################################################
    def _potential_sum(self, output, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_fcns=[], pt_wise_fcns=[]):
        r"""Integrates various energies over the object. Pt-wise or deformation gradient-wise.

        Args:
            output (torch.Tensor): Either a scalar, or vector or hessian depending on the functions.
            z (torch.Tensor): Transformations dofs
            z_dot (torch.Tensor): Time derivative of transformations
            B (torch.Tensor): Precomputed jacobian of flattened pts w.r.t flattened transforms
            dFdz (torch.Tensor): Precomputed jacobian of flattened defo-grad w.r.t. flattened transforms
            x0_flat (torch.Tensor): Flattened sample points at rest state.
            bigI (torch.Tensor): Large Identity matrix. Precomputed up-front
            defo_grad_fcns (list, optional): A list of functions that compute energies, 
                                            or gradients or hessians over the object's sample pts by deformation-gradients. Defaults to [].
            pt_wise_fcns (list, optional): A list of functions that compute energies, 
                                            or gradients or hessians over the object's sample pts by deformed points. Defaults to [].
        """
        # updates the quantity calculated in the output value
        F_ele = torch.matmul(dFdz, z) + bigI
        x_flat = B @ z + x0_flat
        x = x_flat.reshape(-1, 3)
        for e in defo_grad_fcns:
            output += e(F_ele)
        for e in pt_wise_fcns:
            output += e(x)

    def _newton_E(self, z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, defo_grad_energies=[], pt_wise_energies=[]):
        r"""Backward's euler energy used in newton's method

        Args:
            z (torch.Tensor): Transforms
            z_prev (torch.Tensor): Previous transforms
            z_dot (torch.Tensor): Time derivative of transforms
            B (torch.Tensor): Precomputed Jacobian, dx/dz
            BMB (torch.Tensor): Precomputed z-wise mass matrix.
            dt (float): Timestep
            x0_flat (torch.Tensor): Rest state flattened points
            dFdz (torch.Tensor): Precomputed jacobian
            bigI (torch.Tensor): Precomputed identity matrix.
            defo_grad_energies (list, optional): List of energies. Defaults to [].
            pt_wise_energies (list, optional): List of energies. Defaults to [].

        Returns:
            torch.Tensor: Backward's euler energy, single value tensor
        """
        pe_sum = torch.tensor([0], device=self.default_device, dtype=self.default_dtype)
        self._potential_sum(pe_sum, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_energies, pt_wise_energies)
        return 0.5 * z.T @ BMB @ z - z.T @ BMB @ z_prev - dt * z.T @ BMB @ z_dot + dt * dt * pe_sum

    def _newton_G(self, z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, defo_grad_gradients=[], pt_wise_gradients=[]):
        r"""Backward's euler gradient used in newton's method

        Args:
            z (torch.Tensor): Transforms
            z_prev (torch.Tensor): Previous transforms
            z_dot (torch.Tensor): Time derivative of transforms
            B (torch.Tensor): Precomputed Jacobian, dx/dz
            BMB (torch.Tensor): Precomputed z-wise mass matrix.
            dt (float): Timestep
            x0_flat (torch.Tensor): Rest state flattened points
            dFdz (torch.Tensor): Precomputed jacobian
            bigI (torch.Tensor): Precomputed identity matrix.
            defo_grad_gradients (list, optional): List of gradients. Defaults to [].
            pt_wise_gradients (list, optional): List of gradients. Defaults to [].

        Returns:
            torch.Tensor: Backward's euler gradient
        """
        pe_grad_sum = torch.zeros_like(z)
        self._potential_sum(pe_grad_sum, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_gradients, pt_wise_gradients)
        return BMB @ z - BMB @ z_prev - dt * BMB @ z_dot + dt * dt * pe_grad_sum

    def _newton_H(self, z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, defo_grad_hessians=[], pt_wise_hessians=[]):
        r"""Backward's euler hessian used in newton's method

        Args:
            z (torch.Tensor): Transforms
            z_prev (torch.Tensor): Previous transforms. Unused.
            z_dot (torch.Tensor): Time derivative of transforms
            B (torch.Tensor): Precomputed Jacobian, :math:`\frac{dx}{dz}`
            BMB (torch.Tensor): Precomputed z-wise mass matrix.
            dt (float): Timestep
            x0_flat (torch.Tensor): Rest state flattened points
            dFdz (torch.Tensor): Precomputed jacobian
            bigI (torch.Tensor): Precomputed identity matrix.
            defo_grad_hessians (list, optional): List of hessians. Defaults to [].
            pt_wise_hessians (list, optional): List of hessians. Defaults to [].

        Returns:
            torch.Tensor: Backward's euler hessian
        """
        pe_hess_sum = torch.zeros(z.shape[0], z.shape[0], device=self.default_device, dtype=self.default_dtype)
        self._potential_sum(pe_hess_sum, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_hessians, pt_wise_hessians)
        return BMB + dt * dt * pe_hess_sum
    ###########################################################

    def run_sim_step(self):
        r"""Runs one simulation step
        """

        # Get energies
        for idx in self.sim_obj_dict:
            o = self.sim_obj_dict[idx]
            pt_names = list(o.force_dict["pt_wise"].keys())
            pt_wise_energies = [o.force_dict["pt_wise"][name]["energy"] for name in pt_names]
            defo_names = list(o.force_dict["defo_grad_wise"].keys())
            defo_wise_energies = [o.force_dict["defo_grad_wise"][name]["energy"] for name in defo_names]

            o.z_prev = o.z.clone().detach()
            more_partial__newton_E = partial(self._newton_E, z_prev=o.z_prev, z_dot=o.z_dot, B=o.B, BMB=o.BMB, dt=self.timestep, x0_flat=o.x0_flat, dFdz=o.dFdz,
                                             bigI=o.bigI, defo_grad_energies=o.get_all_force_fcns("defo_grad_wise", "energy"), pt_wise_energies=o.get_all_force_fcns("pt_wise", "energy"))
            more_partial__newton_G = partial(self._newton_G, z_prev=o.z_prev, z_dot=o.z_dot, B=o.B, BMB=o.BMB, dt=self.timestep, x0_flat=o.x0_flat, dFdz=o.dFdz,
                                             bigI=o.bigI, defo_grad_gradients=o.get_all_force_fcns("defo_grad_wise", "gradient"), pt_wise_gradients=o.get_all_force_fcns("pt_wise", "gradient"))
            more_partial__newton_H = partial(self._newton_H, z_prev=o.z_prev, z_dot=o.z_dot, B=o.B, BMB=o.BMB, dt=self.timestep, x0_flat=o.x0_flat, dFdz=o.dFdz,
                                             bigI=o.bigI, defo_grad_hessians=o.get_all_force_fcns("defo_grad_wise", "hessian"), pt_wise_hessians=o.get_all_force_fcns("pt_wise", "hessian"))
            o.z = optimization.newtons_method(o.z, more_partial__newton_E, more_partial__newton_G,
                                              more_partial__newton_H, max_iters=self.max_newton_steps, conv_criteria=0)

            F_ele = torch.matmul(o.dFdz, o.z) + o.bigI
            x_flat = o.B @ o.z + o.x0_flat
            x = x_flat.reshape(-1, 3)
            newline = "\t"
            logger.debug(f'\t{newline.join(f"pt-wise {name}: {en(x):8.3} " for name, en in zip(pt_names, pt_wise_energies))}' +
                         f'\t\t{newline.join(f" F-wise {name}:{en(F_ele):8.3} " for name, en in zip(defo_names, defo_wise_energies))}')

            with torch.no_grad():
                o.z_dot = (o.z - o.z_prev) / self.timestep

        self.current_sim_step += 1
