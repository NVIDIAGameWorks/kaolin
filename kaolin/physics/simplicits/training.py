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

from abc import ABC, abstractmethod
import logging
import warnings
import tqdm
from typing import Protocol, runtime_checkable, Any, Optional, Sequence

import torch
import kaolin

from .losses import compute_losses
from .network import SimplicitsMLP, SkinningModule
from kaolin.rep import TensorContainerBase
import kaolin.utils.testing

logger = logging.getLogger(__name__)

__all__ = [
    'SimplicitsObject',
    'PhysicsPoints',
    'SkinnedPoints',
    'SkinnedPhysicsPoints',
]

@runtime_checkable
class PhysicsPointsProtocol(Protocol):
    r"""
    Protocol that gives access to point-sampled object as well as its point-sampled material properties.

    Attributes:
        pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`).
        yms (torch.Tensor): Young's moduli defining material stiffness, of shape :math:`(N,)` (in :math:`kg/m/s^2`).
        prs (torch.Tensor): Poisson's ratios defining material compressibility, of shape :math:`(N,)`.
        rhos (torch.Tensor): Density defining material density, of shape :math:`(N,)` (in :math:`kg/m^3`).
        appx_vol (torch.Tensor): Approximate volume, of shape :math:`(1,)` (in :math:`m^3`).
    """
    pts: torch.Tensor
    yms: torch.Tensor
    prs: torch.Tensor
    rhos: torch.Tensor
    appx_vol: torch.Tensor

    def subsample(self, num_pts=None, sample_indices=None) -> Any:
        ...

class PhysicsPoints(PhysicsPointsProtocol, TensorContainerBase):
    """Point-sampled object with material properties.

    Attributes:
        pts (torch.Tensor): Points tensor representing the object's geometry, of shape :math:`(N, 3)` (in :math:`m`).
        yms (torch.Tensor): Young's moduli defining material stiffness, of shape :math:`(N,)` (in :math:`kg/m/s^2`).
        prs (torch.Tensor): Poisson's ratios defining material compressibility, of shape :math:`(N,)`.
        rhos (torch.Tensor): Density defining material density, of shape :math:`(N,)` (in :math:`kg/m^3`).
        appx_vol (torch.Tensor): Approximate volume, as a scalar (in :math:`m^3`).
    """
    def __init__(self, pts, yms, prs, rhos, appx_vol, strict_checks=True):
        r"""Initialize the class.

        Args:
            pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`)
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness (in :math:`kg/m/s^2`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density (in :math:`kg/m^3`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume (in :math:`m^3`). Can be either:
                - A tensor of shape :math:`(1,)`
                - A float value
        """
        if not torch.is_tensor(yms):
            yms = torch.full((pts.shape[0],), yms, dtype=pts.dtype, device=pts.device)
        else:
            assert yms.shape == (pts.shape[0],), 'yms must be a tensor of shape (N,)'
        if not torch.is_tensor(prs):
            prs = torch.full((pts.shape[0],), prs, dtype=pts.dtype, device=pts.device)
        else:
            assert prs.shape == (pts.shape[0],), 'prs must be a tensor of shape (N,)'
        if not torch.is_tensor(rhos):
            rhos = torch.full((pts.shape[0],), rhos, dtype=pts.dtype, device=pts.device)
        else:
            assert rhos.shape == (pts.shape[0],), 'rhos must be a tensor of shape (N,)'
        if torch.is_tensor(appx_vol):
            appx_vol = appx_vol.squeeze()
        else:
            appx_vol = torch.tensor(appx_vol, dtype=pts.dtype, device=pts.device)

        self.pts = pts
        self.yms = yms
        self.prs = prs
        self.rhos = rhos
        self.appx_vol = appx_vol
        if strict_checks:
            assert self.check_sanity(log_error=True)

    def _get_subsample_indices(self, num_pts=None, sample_indices=None):
        r"""Compute or validate subsample indices.

        Args:
            num_pts (int, optional):
                Number of points to sample. Mutually exclusive with ``sample_indices``.
            sample_indices (torch.Tensor, optional):
                Explicit indices to use for subsampling, of shape :math:`(\text{num_pts},)`.
                Mutually exclusive with ``num_pts``.
        Returns:
            torch.LongTensor: Indices to use for subsampling, of shape :math:`(\text{num_pts},)`.
        """
        if sample_indices is None:
            assert num_pts is not None, 'must specify num_pts or sample_indices'
            if num_pts < self.pts.shape[0]:
                return torch.randperm(len(self.pts), device=self.pts.device)[:num_pts]
            else:
                return torch.arange(len(self.pts), device=self.pts.device)
        else:
            assert num_pts is None or num_pts == -1 or num_pts == sample_indices.shape[0], 'conflicting subsample inputs'
            return sample_indices

    def subsample(self, num_pts=None, sample_indices=None):
        r"""Subsample into another PhysicsPoints.

        Args:
            num_pts (int, optional):
                Number of points to sample. Mutually exclusive with ``sample_indices``.
            sample_indices (torch.Tensor, optional):
                Explicit indices to use for subsampling, of shape :math:`(\text{num_pts},)`.
                Mutually exclusive with ``num_pts``.

        Returns:
            (PhysicsPoints): The subsampled PhysicsPoints, of size :math:`(\text{num_pts})`.
        """
        indices = self._get_subsample_indices(num_pts, sample_indices)

        # TODO: we didn't call float() here; needed?
        pts = self.pts[indices, :]
        rhos = self.rhos[indices]
        yms = self.yms[indices]
        prs = self.prs[indices]
        return PhysicsPoints(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=self.appx_vol)

    def __str__(self):
        r"""String describing object.

        Returns:
            string: String description of object
        """
        unique_densities = torch.unique(self.rhos).tolist()
        unique_yms = torch.unique(self.yms).tolist()
        unique_prs = torch.unique(self.prs).tolist()
        appx_vol = self.appx_vol
        class_name = self.__class__.__name__
        return f"{class_name}(N_points={self.pts.shape[0]}, Densities={unique_densities}, Yms={unique_yms}, Prs={unique_prs}, AppxVol={appx_vol})"

    @classmethod
    def class_tensor_attributes(cls):
        return ["pts", "yms", "prs", "rhos", "appx_vol"]

    @classmethod
    def class_other_attributes(cls):
        return []

    def __len__(self):
        return self.pts.shape[0]

    @property
    def device(self):
        return self.pts.device

    @property
    def dtype(self):
        return self.pts.dtype

    def check_tensor_attribute(self, attr, log_error=False):
        """Checks tensor attribute validity; returns True if valid."""

        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        if attr == "pts":
            expected_shape = (len(self), 3)
        elif attr in ["yms", "prs", "rhos"]:
            expected_shape = (len(self),)
        elif attr == "appx_vol":
            expected_shape = tuple()
        else:
            return False
        try:
            kaolin.utils.testing.check_tensor(getattr(self, attr), shape=expected_shape,
                                              device=self.device, dtype=self.dtype,throw=True)
        except Exception as e:
            _maybe_log(f'Attribute {attr}: {e}')
            return False
        return True


@runtime_checkable
class SkinnedPointsProtocol(Protocol):
    """
    Protocol that gives access to point sampling of an object and its per-point skinning weights.

    This information is sufficient to transform a point-based renderable representation, like Gaussian Splats,
    according to the reduced-order transforms the simulator optimizes.
    """
    pts: torch.Tensor
    skinning_weights: torch.Tensor


class SkinnedPoints(SkinnedPointsProtocol, TensorContainerBase):
    r"""Points with skinning properties.

    Attributes:
        pts (torch.Tensor): The skinned points, of shape :math:`(N, 3)` (in :math:`m`).
        skinning_weights (torch.Tensor): The skinning weights of the points, of shape :math:`(N, \text{num_handles})`.
    """
    def __init__(self, pts, skinning_weights, strict_checks=True):
        self.pts = pts
        self.skinning_weights = skinning_weights
        if strict_checks:
            assert self.check_sanity(log_error=True)

    @property
    def num_handles(self):
        """Last dimension of the skinning weights"""
        return self.skinning_weights.shape[1]

    @classmethod
    def from_skinning_mod(cls, pts: torch.Tensor, skinning_mod: SkinningModule):
        r"""
        Constructor from a :class:`SkinningModule` to be applied on the points.

        Args:
            pts (torch.Tensor): The points to be skinned, of shape :math:`(N, 3)` (in :math:`m`)
            skinning_mod (SkinningModule): The SkinningModule to be used to compute the skinning weights.
        """
        assert isinstance(skinning_mod, SkinningModule), 'skinning_mod must be a SkinningModule'
        skinning_weights = skinning_mod.compute_skinning_weights(pts)
        return cls(pts=pts, skinning_weights=skinning_weights)

    @classmethod
    def class_tensor_attributes(cls):
        return ["pts", "skinning_weights"]

    @classmethod
    def class_other_attributes(cls):
        return []

    def __len__(self):
        return self.pts.shape[0]

    @property
    def device(self):
        return self.pts.device

    @property
    def dtype(self):
        return self.pts.dtype

    def check_tensor_attribute(self, attr, log_error=False):
        """Checks tensor attribute validity; returns True if valid."""

        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        if attr == "pts":
            expected_shape = (len(self), 3)
        elif attr == "skinning_weights":
            expected_shape = (len(self), None)
        else:
            return False
        try:
            kaolin.utils.testing.check_tensor(getattr(self, attr), shape=expected_shape,
                                              device=self.device, dtype=self.dtype,throw=True)
        except Exception as e:
            _maybe_log(f'Attribute {attr}: {e}')
            return False
        return True




@runtime_checkable
class SkinnedPhysicsPointsProtocol(SkinnedPointsProtocol, PhysicsPointsProtocol, Protocol):
    """
    Protocol that gives access to point-sampled object as well as its point-sampled material properties,
    as well as per-point skinning functions and extra matrices needed to simulate the object.

    This information is sufficient for the Simplicits-based simulator to simulate the object
    and optimize the reduced-order transforms according to the energies in the system.
    """
    # Jacobian of skinning weights w.r.t. rest positions, shape (N, num_handles, 3)
    dwdx: Any
    # Optional renderable points (separate from simulation quadrature points)
    renderable: SkinnedPointsProtocol  # Optional[SkinnedPoints]

class SkinnedPhysicsPoints(PhysicsPoints, SkinnedPhysicsPointsProtocol, TensorContainerBase):
    r"""
    Points with skinning properties and material properties.

    Attributes:
        pts (torch.Tensor): The skinned points, of shape :math:`(N, 3)` (in :math:`m`).
        yms (torch.Tensor): The Young's moduli of the points, of shape :math:`(N,)` (in :math:`kg/m/s^2`).
        prs (torch.Tensor): The Poisson's ratios of the points, of shape :math:`(N,)`.
        rhos (torch.Tensor): The densities of the points, of shape :math:`(N,)` (in :math:`kg/m^3`).
        appx_vol (torch.Tensor): The approximate volume of the object, of shape :math:`(1,)` (in :math:`m^3`).
        skinning_weights (torch.Tensor): The skinning weights of the points, of shape :math:`(N, \text{num_handles})`.
        dwdx (torch.Tensor): Jacobian of the skinning weight function w.r.t. rest positions,
            of shape :math:`(N, \text{num_handles}, 3)`.
        renderable (SkinnedPoints): The renderable points, of shape :math:`(\text{num_renderable_pts}, 3)`.
    """
    def __init__(self, pts, yms, prs, rhos, appx_vol, skinning_weights, dwdx,
                 renderable: SkinnedPointsProtocol = None, strict_checks=True):
        r"""
        Constructor for making minimal simulatable data object from points, material properties, and skinning weights.

        Args:
            pts (torch.Tensor): Cubature points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`)
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness (in :math:`kg/m/s^2`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density (in :math:`kg/m^3`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume (in :math:`m^3`). Can be either:
                - A tensor of shape :math:`(1,)`
                - A float value
            skinning_weights (torch.Tensor): Skinning weights tensor of shape :math:`(N, M)` representing the object's skinning weights
            dwdx (torch.Tensor): Jacobian of the skinning weight function w.r.t. rest positions,
                of shape :math:`(N, M, 3)`. Used to build the sparse :math:`dF/dz` matrix inside
                :class:`SimulatedObject`.
            renderable (SkinnedPoints, optional): Skinned points used for rendering
                (e.g. Gaussian splat positions with their skinning weights). Defaults to None.
        """
        super().__init__(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol,
                         strict_checks=False)
        self.skinning_weights = skinning_weights
        self.dwdx = dwdx
        self.renderable = renderable
        if strict_checks:
            assert self.check_sanity(log_error=True)

    @property
    def num_handles(self):
        """Number of skinning handles, including the implicit constant handle."""
        return self.skinning_weights.shape[1]

    def subsample(self, num_pts=None, sample_indices=None):
        r"""Subsample into another SkinnedPhysicsPoints.

        Args:
            num_pts (int, optional):
                Number of points to sample. Mutually exclusive with ``sample_indices``.
            sample_indices (torch.Tensor, optional):
                Explicit indices to use for subsampling, of shape :math:`(\text{num_pts},)`.
                Mutually exclusive with ``num_pts``.
    
        Returns:
            (SkinnedPhysicsPoints): The subsampled SkinnedPhysicsPoints, of size :math:`(\text{num_pts})`.
        """
        indices = self._get_subsample_indices(num_pts, sample_indices)

        # Subsample physics attributes using parent logic
        sampled_points = super().subsample(sample_indices=indices)

        # Subsample skinning-specific attributes
        skinning_weights = self.skinning_weights[indices, :]
        dwdx = self.dwdx[indices]

        return SkinnedPhysicsPoints(
            pts=sampled_points.pts,
            yms=sampled_points.yms,
            prs=sampled_points.prs,
            rhos=sampled_points.rhos,
            appx_vol=sampled_points.appx_vol,
            skinning_weights=skinning_weights,
            dwdx=dwdx
        )

    @classmethod
    def from_skinning_mod(cls, pts, yms, prs, rhos, appx_vol, skinning_mod: SkinningModule, renderable_pts=None):
        r"""
        Constructor from a :class:`SkinningModule` to be applied on the points.

        Args:
            pts (torch.Tensor): The points to be skinned, of shape :math:`(N, 3)` (in :math:`m`)
            yms (torch.Tensor): The Young's moduli of the points, of shape :math:`(N,)` (in :math:`kg/m/s^2`).
            prs (torch.Tensor): The Poisson's ratios of the points, of shape :math:`(N,)`.
            rhos (torch.Tensor): The densities of the points, of shape :math:`(N,)` (in :math:`kg/m^3`).
            appx_vol (torch.Tensor): The approximate volume of the object, of shape :math:`(1,)` (in :math:`m^3`).
            skinning_mod (SkinningModule): The SkinningModule to be used to compute the skinning weights.
            renderable_pts (torch.Tensor, optional): The renderable points, of shape :math:`(\text{num_renderable_pts}, 3)` (in :math:`m`).
                Defaults to None.

        Returns:
            (SkinnedPhysicsPoints): A SkinnedPhysicsPoints with skinning weights and dwdx baked from ``skinning_mod``.
        """
        assert isinstance(skinning_mod, SkinningModule), 'skinning_mod must be a SkinningModule'
        skinning_weights = skinning_mod.compute_skinning_weights(pts)
        dwdx = skinning_mod.compute_dwdx(pts)
        renderable = SkinnedPoints.from_skinning_mod(pts=renderable_pts, skinning_mod=skinning_mod) if renderable_pts is not None else None
        return cls(
            pts=pts,
            yms=yms,
            prs=prs,
            rhos=rhos,
            appx_vol=appx_vol,
            skinning_weights=skinning_weights,
            dwdx=dwdx,
            renderable=renderable
        )
    
    @classmethod
    def class_tensor_attributes(cls):
        return ["pts", "yms", "prs", "rhos", "appx_vol", "skinning_weights", "dwdx", "renderable"]

    @classmethod
    def class_other_attributes(cls):
        return []

    def __len__(self):
        return self.pts.shape[0]

    def check_tensor_attribute(self, attr, log_error=False):
        """Checks tensor attribute validity; returns True if valid."""

        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        if attr == "pts":
            expected_shape = (len(self), 3)
        elif attr in ["yms", "prs", "rhos"]:
            expected_shape = (len(self),)
        elif attr == "appx_vol":
            expected_shape = tuple()
        elif attr == "skinning_weights":
            expected_shape = (len(self), None)
        elif attr == "dwdx":
            expected_shape = (len(self), None, 3)
        elif attr == "renderable":
            if self.renderable is None:
                return True
            return self.renderable.check_sanity(log_error=log_error)
        else:
            return False
        try:
            kaolin.utils.testing.check_tensor(getattr(self, attr), shape=expected_shape,
                                              device=self.device, dtype=self.dtype,throw=True)
        except Exception as e:
            _maybe_log(f'Attribute {attr}: {e}')
            return False
        return True

class SimplicitsObject(PhysicsPoints):
    def __init__(self, pts, yms, prs, rhos, appx_vol, skinning_mod: SkinningModule):
        r"""Initialize a SimplicitsObject with geometry, material properties, and skinning weights.

        A SimplicitsObject is a collection of points, material properties, and a linear blend skinning
        weight function that can be used to deform the object. Objects can be initialized in several ways
        using the static factory methods (read their docstrings for more details). Objects can also be
        denoted as kinematic or dynamic (default). Kinematic objects still have handles, but they are
        not solved for during simulation.

        Args:
            pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`)
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness (in :math:`kg/m/s^2`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density (in :math:`kg/m^3`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float]): Approximate volume (in :math:`m^3`). Can be either:
                - A tensor of shape :math:`(1,)` or :math:`(0,)`
                - A float value
            skinning_mod (SkinningModule):
                SkinningModule to be used to compute the skinning weights.
        """
        super().__init__(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)
        assert isinstance(skinning_mod, SkinningModule), 'skinning_mod must be a SkinningModule'
        self.skinning_mod = skinning_mod
        self.num_handles = self.skinning_mod.compute_skinning_weights(self.pts[:1]).shape[1]

    def to(self, *args: Any, attributes: Optional[Sequence[str]] = None, **kwargs: Any):
        """Moves or casts tensors like :meth:`torch.Tensor.to` / :meth:`torch.nn.Module.to`.

        Args:
            *args: forwarded to ``tensor.to(*args)``
            attributes (list of str, optional): if set, only these tensor attributes are updated
            **kwargs: forwarded to ``tensor.to(**kwargs)``

        Returns:
            PointSamples: shallow copy with converted tensors
        """
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True) + ["skinning_mod"]
        return self._construct_apply(lambda t: t.to(*args, **kwargs), attributes)

    @classmethod
    def create_rigid(cls, pts, yms, prs, rhos, appx_vol=1):
        r"""Creates a rigid SimplicitsObject with a single weight for affine deformations.

        This method creates a SimplicitsObject that behaves as a rigid body. At low stiffness values
        (young's modulus/ym), deformations will not be expressive, but with high stiffness values,
        the object will act as rigid.

        Args:
            pts (torch.Tensor): Points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`)
            yms (Union[torch.Tensor, float]): Young's moduli defining material stiffness (in :math:`kg/m/s^2`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            prs (Union[torch.Tensor, float]): Poisson's ratios defining material compressibility. Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            rhos (Union[torch.Tensor, float]): Density defining material density (in :math:`kg/m^3`). Can be either:
                - A tensor of shape :math:`(N,)` for per-point values
                - A float value that will be applied to all points
            appx_vol (Union[torch.Tensor, float], optional): Approximate volume (in :math:`m^3`). Can be either:
                - A tensor of shape :math:`(1,)` or :math:`(0,)`
                - A float value. Defaults to 1

        Returns:
            (SimplicitsObject): A rigid SimplicitsObject with a constant weight function.
        """
        return cls(
            pts=pts,
            yms=yms,
            prs=prs,
            rhos=rhos,
            appx_vol=appx_vol,
            skinning_mod=SkinningModule.from_function(lambda x: torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype))
        )

    @classmethod
    def create_trained(cls, pts=None, yms=None, prs=None, rhos=None, appx_vol=None, physics_points=None,
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
                       normalize_for_training=True,
                       display_progress=False):
        r"""Constructs a SimplicitsObject by training a neural network to learn skinning weights.

        This method creates a SimplicitsObject by training a neural network to learn skinning weights
        that can be used for deformation. The network is trained to minimize a combination of
        local and global energy terms.
        
        Note:
            If num_handles is set to 1, the object will be created as rigid instead of deformable.
            The training process uses a combination of local and global energy terms to ensure
            both local detail preservation and global shape maintenance.

        Args:
            physics_points (PhysicsPoints, optional): PhysicsPoints object to be used for training. Defaults to None.
                If provided, ``pts``, ``yms``, ``prs``, ``rhos``, and ``appx_vol`` must all be ``None`` and the values
                are taken from this object instead.
            pts (torch.Tensor, optional): Deprecated, use ``physics_points`` instead.
                Points tensor of shape :math:`(N, 3)` representing the object's geometry (in :math:`m`).
                Required unless ``physics_points`` is provided.
            yms (Union[torch.Tensor, float], optional): Deprecated, use ``physics_points`` instead.
                Young's moduli defining material stiffness (in :math:`kg/m/s^2`); either a tensor of shape :math:`(N,)`
                for per-point values, or a float value applied to all points.
                Required unless ``physics_points`` is provided.
            prs (Union[torch.Tensor, float], optional): Deprecated, use ``physics_points`` instead.
                Poisson's ratios defining material compressibility; either a tensor of shape :math:`(N,)`
                for per-point values, or a float value applied to all points.
                Required unless ``physics_points`` is provided.
            rhos (Union[torch.Tensor, float], optional): Deprecated, use ``physics_points`` instead.
                Density defining material density (in :math:`kg/m^3`); either a tensor of shape :math:`(N,)`
                for per-point values, or a float value applied to all points.
                Required unless ``physics_points`` is provided.
            appx_vol (Union[torch.Tensor, float], optional): Deprecated, use ``physics_points`` instead.
                Approximate volume (in :math:`m^3`); either a tensor of shape :math:`(1,)` or :math:`(0,)`, or a float value.
                Required unless ``physics_points`` is provided.
            num_handles (int, optional): Number of control handles for deformation. Defaults to 10
            num_samples (int, optional): Number of samples used for training. Defaults to 1000
            model_layers (int, optional): Number of layers in the neural network. Defaults to 6
            training_batch_size (int, optional): Batch size for training. Defaults to 10
            training_num_steps (int, optional): Number of training iterations. Defaults to 10000.
            training_lr_start (float, optional): Starting learning rate. Defaults to 1e-3.
            training_lr_end (float, optional): Final learning rate. Defaults to 1e-3
            training_le_coeff (float, optional): Coefficient for local energy term. Defaults to 1e-1
            training_lo_coeff (float, optional): Coefficient for global energy term. Defaults to 1e6
            training_log_every (int, optional): Logging frequency during training. Defaults to 1000
            normalize_for_training (bool, optional): Whether to normalize points to unit cube for training. Defaults to True
            display_progress (bool, optional): If True, display a tqdm progress bar during training. Defaults to False.

        Returns:
            (SimplicitsObject): A trained SimplicitsObject with learned skinning weights.

        """
        if physics_points is not None:
            assert pts is None and yms is None and prs is None and rhos is None and appx_vol is None, 'pts, yms, prs, rhos, and appx_vol must be None if physics_points is provided'
            pts = physics_points.pts
            yms = physics_points.yms
            prs = physics_points.prs
            rhos = physics_points.rhos
            appx_vol = physics_points.appx_vol
        else:
            warnings.warn('pts, yms, prs, rhos, and appx_vol arguments are deprecated. Please use physics_points instead.', UserWarning, stacklevel=2)
        assert num_handles >= 1, 'Number of handles must be greater or equal than 1'
        if num_handles == 1:
            warnings.warn('Num Handles is 1. Simplicits Object will be created as rigid.',
                          UserWarning, stacklevel=2)

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

        # normalize the points
        if (normalize_for_training):
            bb_max = torch.max(pts, dim=0).values
            bb_min = torch.min(pts, dim=0).values
            bb_vol = (bb_max[0] - bb_min[0]) * (bb_max[1] - bb_min[1]) * (bb_max[2] - bb_min[2])

            # Set pts and appx_vol to normalized values
            training_pts = (pts - bb_min) / (bb_max - bb_min)
            training_appx_vol = appx_vol / bb_vol
        else:
            bb_max = 1
            bb_min = 0
            training_pts = pts
            training_appx_vol = appx_vol

        training_yms = yms.unsqueeze(-1)
        training_prs = prs.unsqueeze(-1)
        training_rhos = rhos.unsqueeze(-1)

        ######### Train the model #########
        model = SimplicitsMLP(3, 64, num_handles, model_layers, bb_min=bb_min, bb_max=bb_max)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), training_lr_start)

        model.train()
        e_logs = []
        for i in tqdm.trange(training_num_steps, desc='Training SimplicitsObject', disable=not display_progress):
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
        return SimplicitsObject(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol, skinning_mod=model)

    def subsample(self, num_pts=None, sample_indices=None):
        r"""Subsample into another SimplicitsObject sharing the same skinning module.

        Args:
            num_pts (int, optional):
                Number of points to sample. Mutually exclusive with ``sample_indices``.
            sample_indices (torch.Tensor, optional):
                Explicit indices to use for subsampling, of shape :math:`(\text{num_pts},)`.
                Mutually exclusive with ``num_pts``.

        Returns:
            (SimplicitsObject): The subsampled SimplicitsObject with the same ``skinning_mod``.
        """
        indices = self._get_subsample_indices(num_pts, sample_indices)

        # Subsample physics attributes using parent logic
        sampled_points = super().subsample(sample_indices=indices)

        # Return a new SimplicitsObject with the same skinning_mod
        # (the function works on any points, so no need to modify it)
        return SimplicitsObject(
            pts=sampled_points.pts,
            yms=sampled_points.yms,
            prs=sampled_points.prs,
            rhos=sampled_points.rhos,
            appx_vol=sampled_points.appx_vol,
            skinning_mod=self.skinning_mod,
        )

    def bake(self, num_qps=None, sampling_indices=None, renderable_pts=None) -> SkinnedPhysicsPoints:
        r"""
        Bakes the skinning weights for simulation, and optionally also bakes renderable points.

        Produces a SkinnedPhysicsPoints object ready for use in a SimplicitsScene. If
        renderable_pts is provided, those points are also skinned and stored in the returned
        object for later use in rendering queries.

        Args:
            num_qps (int, optional):
                Number of quadrature points to sample. Mutually exclusive with ``sampling_indices``.
            sampling_indices (torch.Tensor, optional):
                Explicit quadrature point indices to use, of shape :math:`(\text{num_qps},)`.
                Mutually exclusive with ``num_qps``.
            renderable_pts (torch.Tensor, optional): Additional points (e.g. Gaussian splat
                positions, in :math:`m`) whose skinning weights should be baked for rendering. Defaults to None.

        Returns:
            (SkinnedPhysicsPoints): Baked object suitable for simulation.
        """
        if num_qps is None and sampling_indices is None:
            raise ValueError(
                "bake() requires either num_qps or sampling_indices to be specified."
            )
        # TODO: What if collisions need separated points
        sampled = self.subsample(num_pts=num_qps, sample_indices=sampling_indices)
        return SkinnedPhysicsPoints.from_skinning_mod(
            pts=sampled.pts,
            yms=sampled.yms,
            prs=sampled.prs,
            rhos=sampled.rhos,
            appx_vol=sampled.appx_vol,
            skinning_mod=self.skinning_mod,
            renderable_pts=renderable_pts
        )

    def bake_for_rendering(self, renderable_pts) -> SkinnedPoints:
        r"""
        Bakes the skinning weights for rendering.

        Args:
            renderable_pts (torch.Tensor): Additional points (e.g. Gaussian splat
                positions) whose skinning weights should be baked for rendering,
                of shape :math:`(\text{num_renderable_pts}, 3)` (in :math:`m`).

        Returns:
            (SkinnedPoints): Baked SkinnedPoints for renderable.
        """
        return SkinnedPoints.from_skinning_mod(pts=renderable_pts, skinning_mod=self.skinning_mod)
