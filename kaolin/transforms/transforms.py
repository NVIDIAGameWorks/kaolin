# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from typing import Iterable, List, Optional, Type, Union, Callable
from pathlib import Path
import hashlib

import scipy
import numpy as np
import torch

from kaolin.rep.PointCloud import PointCloud
from kaolin.rep.VoxelGrid import VoxelGrid
from kaolin.rep.Mesh import Mesh
from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.rep.QuadMesh import QuadMesh

import kaolin.conversions as cvt

# from kaolin.conversion import mesh as cvt_mesh
# from kaolin.conversion import SDF as cvt_SDF
# from kaolin.conversion import voxel as cvt_voxel
from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import meshfunc
from kaolin.transforms import voxelfunc


class Compose(object):
    """Composes (chains) multiple transforms sequentially. Identical to
    `torchvision.transforms.Compose`.

    Args:
        tforms (list): List of transforms to compose.

    TODO: Example.

    """

    def __init__(self, transforms: Iterable):
        self.tforms = transforms

    def __call__(self, inp: torch.Tensor):
        for t in self.tforms:
            inp = t(inp)
        return inp

    def __repr__(self):
        fstr = self.__class__.__name__ + '('
        for t in self.tforms:
            fstr += '\n'
            fstr += ' {0}'.format(t)
        fstr += '\n)'
        return fstr


class CacheCompose(object):
    """Caches the results of the provided compose pipeline to disk.
    If the pipeline is already cached, data is returned from disk,
    otherwise, data is converted following the provided transforms.

        Args:
            transforms (Iterable): List of transforms to compose.
            cache_dir (str): Directory where objects will be cached. Default
                             to 'cache'.
    """

    def __init__(self, transforms: Iterable, cache_dir: str = 'cache'):
        self.compose = Compose(transforms)
        self.cache_dir = Path(cache_dir) / self.get_hash()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_ids = [p.stem for p in self.cache_dir.glob('*')]

    def __call__(self, object_id: str, inp: Union[torch.Tensor, Mesh] = None):
        """Transform input. If transformed input was cached, is is read from disk

            Args:
                inp (torch.Tensor or Mesh): input tensor or Mesh object to be transformed,
                name (str): object name used to write and read from disk.

            Returns:
                Union[torch.Tensor, Mesh]: Tensor or Mesh object.
        """

        fpath = self.cache_dir / '{0}.npz'.format(object_id)

        if not fpath.exists():
            assert inp is not None
            transformed = self.compose(inp)
            self._write(transformed, fpath)
            self.cached_ids.append(object_id)
        else:
            transformed = self._read(fpath)

        return transformed

    def _write(self, x, fpath):
        if isinstance(x, Mesh):
            np.savez(fpath, vertices=x.vertices.data.cpu().numpy(),
                     faces=x.faces.data.cpu().numpy())
        elif isinstance(x, VoxelGrid):
            # Save voxel grid as sparse matrix for quick loading
            res = x.voxels.size(0)
            sparse_voxel = scipy.sparse.csc_matrix(x.voxels.reshape(res, -1).cpu().numpy())
            # np.savez_compressed(fpath, sparse=sparse_voxel)
            scipy.sparse.save_npz(fpath, sparse_voxel)
        else:
            np.savez(fpath, x.data.cpu().numpy())

    def _read(self, fpath):
        data = np.load(fpath)
        if 'vertices' in data and 'faces' in data:
            verts = torch.from_numpy(data['vertices'])
            faces = torch.from_numpy(data['faces'])
            if data['faces'].shape[-1] == 4:
                data = QuadMesh.from_tensors(verts, faces)
            else:
                data = TriangleMesh.from_tensors(verts, faces)
        elif 'format' in data:
            matrix_format = data['format'].item()
            sparse = scipy.sparse.csc_matrix((data['data'], data['indices'], data['indptr']), data['shape'])
            data = torch.from_numpy(sparse.todense())
            res = data.size(0)
            data = data.reshape(res, res, res)
        else:
            data = torch.from_numpy(data['arr_0'])

        return data

    def get_hash(self):
        return hashlib.md5(bytes(str(self.compose), 'utf-8')).hexdigest()


class NumpyToTensor(object):
    """Converts a `np.ndarray` object to a `torch.Tensor` object. """

    def __call__(self, arr: np.ndarray):
        """
        Args:
            arr (np.ndarray): Numpy array to be converted to Tensor.

        Returns:
            (torch.Tensor): Converted array
        """
        return torch.from_numpy(arr)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ScalePointCloud(object):
    """Scale a pointcloud with a fixed scaling factor.
    Given a scale factor `scf`, this transform will scale each point in the
    pointcloud, i.e.,
    ``cloud = scf * cloud``

    Args:
        scf (int or float or torch.Tensor): Scale factor by which input clouds
            are to be scaled (Note: if passing in a torch.Tensor type, only
            one-element tensors are allowed).
        inplace (bool, optional): Whether or not the transformation should be
            in-place (default: True).

    TODO: Example.

    """

    def __init__(self, scf: Union[int, float, torch.Tensor],
                 inplace: Optional[bool] = True):
        self.scf = scf
        self.inplace = inplace

    def __call__(self, cloud: Union[torch.Tensor, PointCloud]):
        """
        Args:
            cloud (torch.Tensor or PointCloud): Pointcloud to be scaled.

        Returns:
            (torch.Tensor or PointCloud): Scaled pointcloud.
        """
        return pcfunc.scale(cloud, scf=self.scf, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(scf={0})'.format(self.scf)


class RotatePointCloud(object):
    r"""Rotate a pointcloud with a given rotation matrix.
    Given a :math:`3 \times 3` rotation matrix, this transform will rotate each
    point in the cloud by the rotation matrix specified.

    Args:
        rotmat (torch.Tensor): Rotation matrix that specifies the rotation to
            be applied to the pointcloud (shape: :math:`3 \times 3`).
        inplace (bool, optional): Bool to make this operation in-place.

    TODO: Example.

    """

    def __init__(self, rotmat: torch.Tensor, inplace: Optional[bool] = True):
        self.rotmat = rotmat
        self.inplace = inplace

    def __call__(self, cloud: Union[torch.Tensor, PointCloud]):
        """
        Args:
            cloud (torch.Tensor or PointCloud): Input pointcloud to be rotated.

        Returns:
            (torch.Tensor or PointCloud): Rotated pointcloud.
        """
        return pcfunc.rotate(cloud, rotmat=self.rotmat, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(rotmat={0})'.format(self.rotmat)


class RealignPointCloud(object):
    r"""Re-align a `src` pointcloud such that it fits in an axis-aligned
    bounding box whose size matches the `tgt` pointcloud.

    Args:
        tgt (torch.Tensor or PointCloud): Target pointcloud, to whose
            dimensions the source pointcloud must be aligned.
        inplace (bool, optional): Bool to make this operation in-place.

    TODO: Example.

    """

    def __init__(self, tgt: Union[torch.Tensor, PointCloud],
                 inplace: Optional[bool] = True):
        self.tgt = tgt
        self.inplace = inplace

    def __call__(self, src: Union[torch.Tensor, PointCloud]):
        """
        Args:
            src (torch.Tensor or PointCloud): Source pointcloud, which needs
                to be aligned to the target pointcloud.

        Returns:
            (torch.Tensor or PointCloud): Source pointcloud aligned to match
                the axis-aligned bounding box of the target pointcloud `tgt`.
        """
        return pcfunc.realign(src, self.tgt, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(realign)'


class NormalizePointCloud(object):
    r"""Normalize a pointcloud such that it is centered at the orgin and has
    unit standard deviation.

    Args:
        inplace (bool, optional): Bool to make this operation in-place.

    TODO: Example.

    """

    def __init__(self, inplace: Optional[bool] = True):
        self.inplace = inplace

    def __call__(self, cloud: Union[torch.Tensor, PointCloud]):
        r"""
        Args:
            src (torch.Tensor or PointCloud): Pointcloud to be normalized
                (shape: :math:`B \times \cdots \times N \times D`, where
                :math:`B` is the batchsize (optional), :math:`N` is the
                number of points in the cloud, and :math:`D` is the
                dimensionality of the cloud.
        """
        return pcfunc.normalize(cloud, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(normalize)'


class DownsampleVoxelGrid(object):
    r"""Downsamples a voxelgrid, given a (down)scaling factor for each
    dimension.

    .. Note::
        The voxel output is not thresholded.

    Args:
        scale (list): List of tensors to scale each dimension down by
            (length: 3).
        inplace (bool, optional): Bool to make the operation in-place.

    TODO: Example.

    """

    def __init__(self, scale: List[int], inplace=True):
        self.scale = scale
        self.inplace = inplace

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxgrid (torch.Tensor or VoxelGrid): Voxel grid to be downsampled
                (shape: must be a tensor containing exactly 3 dimensions).

        Returns:
            (torch.Tensor): Downsampled voxel grid.
        """
        return cvt.downsample(voxgrid, scale=self.scale,
                              inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(downsample={0})'.format(self.scale)


class UpsampleVoxelGrid(object):
    r"""Upsamples a voxelgrid, given a target dimensionality (this target
    dimensionality is homogeneously applied to all three axes).

    .. Note::
        The output voxels are not thresholded to contain values in the range
        [0, 1].

    Args:
        dim (int): New dimensionality (number of voxels along each dimension
            in the resulting voxel grid).

    TODO: Example.

    """

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxgrid (torch.Tensor or VoxelGrid): Voxel grid to be upsampled
                (shape: must be a tensor containing exactly 3 dimensions).

        Returns:
            (torch.Tensor): Upsampled voxel grid.
        """
        return cvt.upsample(voxgrid, dim=self.dim)

    def __repr__(self):
        return self.__class__.__name__ + '(upsample={0})'.format(self.dim)


class ThresholdVoxelGrid(object):
    r"""Binarizes the voxel array using a specified threshold.

    Args:
        thresh (float): Threshold with which to binarize.
        inplace (bool, optional): Bool to make the operation in-place.

    """

    def __init__(self, thresh: float, inplace: Optional[bool] = True):
        self.thresh = thresh
        self.inplace = inplace

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel array to be binarized.

        Returns:
            (torch.Tensor): Thresholded voxel array.
        """
        return cvt.threshold(voxgrid, thresh=self.thresh,
            inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0})'.format(self.thresh)


class FillVoxelGrid(object):
    r"""Fills the internal structures in a voxel grid. Used to fill holds
    and 'solidify' objects.

    Args:
        thresh (float): Threshold to use for binarization of the grid.

    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor or VoxelGrid): Voxel grid to be filled.

        Returns:
            (torch.Tensor): Filled-in voxel grid.
        """
        return cvt.fill(voxgrid, thresh=self.thresh)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={0})'.format(self.thresh)


class ExtractSurfaceVoxels(object):
    r"""Removes any inernal structure(s) from a voxel array.

    Args:
        thresh (float): threshold with which to binarize
    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel grid from which to extract surface.

        Returns:
            (torch.Tensor): Voxel grid with the internals removed (i.e.,
                containing only voxels that reside on the surface of the
                object).
        """
        return cvt.extract_surface(voxgrid, self.thresh)

    def __repr__(self):
        return self.__class__.__name__ + '(extract_surface={0})'.format(
            self.thresh)


class ExtractOdmsFromVoxelGrid(object):
    r"""Extracts a set of orthographic depth maps from a voxel grid.
    """

    def __init__(self):
        pass

    def __call__(self, voxgrid: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor or VoxelGrid): Voxel grid from which ODMs are
                extracted.

        Returns:
            (torch.Tensor): 6 ODMs from the 6 primary viewing angles.
        """
        return cvt.extract_odms(voxgrid)

    def __repr__(self):
        return self.__class__.__name__ + '(extract_odms)'


class ExtractProjectOdmsFromVoxelGrid(object):
    r"""Extracts a set of orthographic depth maps (odms) from a voxel grid and
        then projects the odms onto a voxel grid.
    """

    def __init__(self):
        pass

    def __call__(self, voxel: Union[torch.Tensor, VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor or VoxelGrid): Voxel grid from which ODMs are
                extracted.

        Returns:
            (torch.Tensor): Voxel grid.
        """
        odms = cvt.extract_odms(voxel)
        return VoxelGrid(cvt.project_odms(odms))

    def __repr__(self):
        return self.__class__.__name__


class SampleTriangleMesh(object):
    r"""Sample points uniformly over the surface of a triangle mesh.

    Args:
        num_samples (int): Number of points to sample from the mesh.
        eps (float, optional): A small number to prevent division by zero
                     for small surface areas.
    """

    def __init__(self, num_samples: int, eps: Optional[float] = 1e-10):
        self.num_samples = num_samples
        self.eps = eps

    def __call__(self, mesh: TriangleMesh):
        """
        Args:
            mesh (TriangleMesh): A triangle mesh object.

        Returns:
            (torch.Tensor): Uniformly sampled points over the surface of the
                input mesh.
        """
        if not isinstance(mesh, TriangleMesh):
            raise TypeError('Input mesh must be of type TriangleMesh. '
                            'Got {0} instead'.format(type(mesh)))
        return meshfunc.sample_triangle_mesh(mesh.vertices, mesh.faces,
                                             self.num_samples, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(sample_triangle_mesh={0})'.format(
            self.num_samples)


class NormalizeMesh(object):
    r"""Normalize a mesh such that it is centered at the orgin and has
    unit standard deviation.

    Args:
        inplace (bool, optional): Bool to make this operation in-place.

    TODO: Example.

    """

    def __init__(self, inplace: Optional[bool] = True):
        self.inplace = inplace

    def __call__(self, mesh: Type[Mesh]):
        r"""
        Args:
            mesh (Mesh): Mesh to be normalized.

        Returns:
            (Mesh): Normalized mesh (centered at origin,
                unit variance along all dimensions)
        """
        return meshfunc.normalize(mesh, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(normalize)'


class ScaleMesh(object):
    r"""Scale a mesh given a specified scaling factor. A scalar scaling factor
    can be provided, in which case it is applied isotropically to all dims.
    Optionally, a list/tuple of anisotropic scale factors can be provided per
    dimension.

    Args:
        scf (float or iterable): Scaling factor per dimension. If only a single
            scaling factor is provided (or a list of size 1 is provided), it is
            isotropically applied to all dimensions. Else, a list/tuple of 3
            scaling factors is expected, which are applied to the X, Y, and Z
            directions respectively.
        inplace (bool, optional): Bool to make this operation in-place.

    """

    def __init__(self, scf: Union[float, int, Iterable],
                 inplace: Optional[bool] = True):
        self.scf = scf
        self.inplace = inplace

    def __call__(self, mesh: Type[Mesh]):
        """
        Args:
            mesh (Mesh): Mesh to be scaled.

        Returns:
            (Mesh): Scaled mesh.
        """
        return meshfunc.scale(mesh, scf=self.scf, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(scf={0})'.format(self.scf)


class TranslateMesh(object):
    r"""Translate a mesh given a (3D) translation vector.

    Args:
        trans (torch.Tensor or iterable): Translation vector (shape:
            torch.Tensor or iterable must have exactly 3 elements).
        inplace (bool, optional): Bool to make this operation in-place.
    """

    def __init__(self, trans: Union[torch.Tensor, Iterable],
                 inplace: Optional[bool] = True):
        self.trans = trans
        self.inplace = inplace

    def __call__(self, mesh: Type[Mesh]):
        """
        Args:
            mesh (Mesh): Mesh to be translated.

        Returns:
            (Mesh): Translated mesh.
        """
        return meshfunc.translate(mesh, trans=self.trans, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(trans={0})'.format(self.trans)


class RotateMesh(object):
    r"""Rotate a mesh given a 3 x 3 rotation matrix.

    Args:
        rotmat (torch.Tensor): Rotation matrix (shape: :math:`3 \times 3`).
        inplace (bool, optional): Bool to make this operation in-place.
    """

    def __init__(self, rotmat: torch.Tensor, inplace: Optional[bool] = True):
        self.rotmat = rotmat
        self.inplace = inplace

    def __call__(self, mesh: Type[Mesh]):
        """
        Args:
            mesh (Mesh): Mesh to be rotated.

        Returns:
            (Mesh): Rotated mesh.
        """
        return meshfunc.rotate(mesh, rotmat=self.rotmat, inplace=self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(rotmat={0})'.format(self.rotmat)


class TriangleMeshToPointCloud(object):
    r"""Converts a triange mesh to a pointcloud with a specified number of
    points. Uniformly samples points over the surface of the mesh.

    Args:
        num_samples (int): Number of points to sample from the mesh.
        eps (float, optional): A small number to prevent division by zero
                     for small surface areas.
    """

    def __init__(self, num_samples: int, eps: Optional[float] = 1e-10):
        self.num_samples = num_samples
        self.eps = eps

    def __call__(self, mesh: TriangleMesh):
        """
        Args:
            mesh (TriangleMesh): A triangle mesh object.

        Returns:
            (torch.Tensor): Uniformly sampled points over the surface of the
                input mesh.
        """
        if not isinstance(mesh, TriangleMesh):
            raise TypeError('Input mesh must be of type TriangleMesh. '
                            'Got {0} instead'.format(type(mesh)))
        return meshfunc.sample_triangle_mesh(mesh.vertices, mesh.faces,
                                             self.num_samples, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(num_samples={0})'.format(
            self.num_samples)


class TriangleMeshToVoxelGrid(object):
    r"""Converts a triangle mesh to a voxel grid with a specified reolution.
    The resolution of the voxel grid is assumed to be homogeneous along all
    three dimensions (X, Y, Z axes).

    Args:
        resolution (int): Desired resolution of generated voxel grid.
        normalize (bool): Determines whether to normalize vertices to a
            unit cube centered at the origin.
        vertex_offset (float): Offset applied to all vertices after
                               normalizing.

    """

    def __init__(self, resolution: int,
                 normalize: bool = True,
                 vertex_offset: float = 0.):
        self.resolution = resolution
        self.normalize = normalize
        self.vertex_offset = vertex_offset

    def __call__(self, mesh: TriangleMesh):
        """
        Args:
            mesh (kaolin.rep.TriangleMesh): Triangle mesh to convert to a
                voxel grid.

        Returns:
            voxgrid (kaolin.rep.VoxelGrid): Converted voxel grid.

        """
        voxels = cvt.trianglemesh_to_voxelgrid(mesh, self.resolution,
                                               normalize=self.normalize,
                                               vertex_offset=self.vertex_offset)
        return voxels

    def __repr__(self):
        return self.__class__.__name__ + '(resolution={0}, normalize={1}, vertex_offset={2})'.\
            format(self.resolution, self.normalize, self.vertex_offset)


class TriangleMeshToSDF(object):
    r"""Converts a triangle mesh to a non-parameteric (point-based) signed
    distance function (SDF).

    Args:
        num_samples (int): Number of points to sample on the surface of the
            triangle mesh.
        noise (float): Fraction of distance from the surface from which the
            SDF is sampled (Eg. a value of 0.05 samples points that are at
            a 5% fraction outside/inside the surface).

    """

    def __init__(self, num_samples: int = 10000, noise: float = 0.05):
        self.num_samples = num_samples
        self.noise = 1 + noise

    def __call__(self, mesh: TriangleMesh):
        """
        Args:
            mesh (kaolin.rep.TriangleMesh): Triangle mesh to convert to a
                signed distance function.

        Returns:
            (torch.Tensor): A signed distance function.
        """
        sdf = cvt.trianglemesh_to_sdf(mesh, self.num_samples)
        return sdf(self.noise * (torch.rand(self.num_samples, 3).to(mesh.device) - .5))

    def __repr__(self):
        return self.__class__.__name__ + '(num_samples={0}, noise={1})'.\
            format(self.num_samples, self.noise)


class MeshLaplacianSmoothing(object):
    r""" Applies laplacian smoothing to the mesh.

        Args:
            iterations (int) : number of iterations to run the algorithm for.
    """

    def __init__(self, iterations: int):
        self.iterations = iterations

    def __call__(self, mesh: Type[Mesh]):
        """
        Args:
            mesh (Mesh): Mesh to be smoothed.

        Returns:
            (Mesh): Rotated mesh.
        """
        mesh.laplacian_smoothing(self.iterations)
        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(iterations={0})'.format(self.iterations)


class RealignMesh(object):
    r""" Aligns the vertices to be in the same (axis-aligned) bounding
    box as that of `target` vertices or point cloud.

    Args:
        target (torch.Tensor or PointCloud) : Target pointcloud to which `src`is
            to be transformed (The `src` cloud is transformed to the
            axis-aligned bounding box that the target cloud maps to). This
            cloud must have the same number of dimensions :math:`D` as in the
            source cloud. (shape: :math:`\cdots \times \cdots \times D`).
        inplace (bool, optional): Bool to make the transform in-place.

    Returns:
        (torch.Tensor): Pointcloud `src` realigned to fit in the (axis-aligned)
            bounding box of the `tgt` cloud.
    """

    def __init__(self, target: Union[torch.Tensor, PointCloud]):
        self.target = target

    def __call__(self, mesh: Type[Mesh]):
        """
        Args:
            mesh (Mesh): Mesh to be realigned.

        Returns:
            (Mesh): Realigned mesh.
        """
        mesh.vertices = pcfunc.realign(mesh.vertices, self.target)
        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(realign)'


class SDFToTriangleMesh(object):
    r""" Converts an SDF function to a mesh

    Args:
        bbox_center (float): Center of the surface's bounding box.
        bbox_dim (float): Largest dimension of the surface's bounding box.
        resolution (int) : The initial resolution of the voxel, should be large enough to
            properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will be doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps).
    """

    def __init__(self, bbox_center: float, bbox_dim: float, resolution: int, upsampling_steps: int):
        self.bbox_center = bbox_center
        self.bbox_dim = bbox_dim
        self.resolution = resolution
        self.upsampling_steps = upsampling_steps

    def __call__(self, sdf: Callable):
        """
        Args:
            sdf (Callable): An object with a .eval_occ function which indicates
                       which of a set of passed points is inside the surface.

        Returns:
            (TriangleMesh): Computed triangle mesh.
        """
        verts, faces = cvt.sdf_to_trianglemesh(sdf, self.bbox_center, self.bbox_dim,
                                       self.resolution, self.upsampling_steps)
        return TriangleMesh.from_tensors(vertices=verts, faces=faces)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(bbox_center={0}'.format(self.bbox_center)
        format_string += ', bbox_dim={0}'.format(self.bbox_dim)
        format_string += ', resolution={0}'.format(self.resolution)
        format_string += ', upsampling_steps={0}'.format(self.upsampling_steps)
        return format_string


class SDFToPointCloud(object):
    r""" Converts an SDF fucntion to a point cloud

    Args:
        bbox_center (float): Center of the surface's bounding box.
        bbox_dim (float): Largest dimension of the surface's bounding box.
        resolution (int) : The initial resolution of the voxel, should be large enough to
            properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will be doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps).
        num_points (int): Number of points in computed point cloud.
    """

    def __init__(self, bbox_center: float, bbox_dim: float, resolution: int,
                 upsampling_steps: int, num_points: int):
        self.bbox_center = bbox_center
        self.bbox_dim = bbox_dim
        self.resolution = resolution
        self.upsampling_steps = upsampling_steps
        self.num_points = num_points

    def __call__(self, sdf: Callable):
        """
        Args:
            sdf (Callable): An object with a .eval_occ fucntion which indicates
                       which of a set of passed points is inside the surface.

        Returns:
            (torch.FloatTensor): Computed point cloud.
        """
        return cvt.sdf_to_pointcloud(sdf, self.bbox_center, self.bbox_dim, self.resolution,
                                self.upsampling_steps, self.num_points)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(bbox_center={0}'.format(self.bbox_center)
        format_string += ', bbox_dim={0}'.format(self.bbox_dim)
        format_string += ', resolution={0}'.format(self.resolution)
        format_string += ', upsampling_steps={0}'.format(self.upsampling_steps)
        format_string += ', num_points={0}'.format(self.num_points)
        return format_string


class SDFToVoxelGrid(object):
    r""" Converts an SDF function to a to a voxel grid

    Args:
        bbox_center (float): Center of the surface's bounding box.
        bbox_dim (float): Largest dimension of the surface's bounding box.
        resolution (int) : The initial resolution of the voxel, should be large enough to
            properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will be doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps).
    """

    def __init__(self, bbox_center: float, bbox_dim: float, resolution: int,
                 upsampling_steps: int, num_points: int):
        self.bbox_center = bbox_center
        self.bbox_dim = bbox_dim
        self.resolution = resolution
        self.upsampling_steps = upsampling_steps
        self.num_points = num_points

    def __call__(self, sdf: Callable):
        """
        Args:
            sdf (Callable): An object with a .eval_occ fucntion which indicates
                       which of a set of passed points is inside the surface.

        Returns:
            (torch.FloatTensor): Computed point cloud.
        """
        return cvt.sdf_to_voxelgrid(sdf, self.bbox_center, self.bbox_dim, self.resolution,
                            self.upsampling_steps, self.num_points)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(bbox_center={0}'.format(self.bbox_center)
        format_string += ', bbox_dim={0}'.format(self.bbox_dim)
        format_string += ', resolution={0}'.format(self.resolution)
        format_string += ', upsampling_steps={0}'.format(self.upsampling_steps)
        return format_string


class VoxelGridToTriangleMesh(object):
    r""" Converts passed voxel to a mesh

    Args:
        thresh (float): threshold from which to make voxel binary
        mode (str):
            -'exact': exect mesh conversion
            -'marching_cubes': marching cubes is applied to passed voxel
        normalize (bool): whether to scale the array to (-.5,.5)
    """
    def __init__(self, threshold, mode, normalize):
        self.thresh = threshold
        self.mode = mode
        self.normalize = normalize

    def __call__(self, voxel: Type[VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel grid.

        Returns:
            (TriangleMesh): Converted triangle mesh.
        """
        verts, faces = cvt.voxelgrid_to_trianglemesh(voxel, self.thresh, self.mode, self.normalize)
        return TriangleMesh.from_tensors(vertices=verts, faces=faces)

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0}, mode={1}, normalize={2})'.\
            format(self.thresh, self.mode, self.normalize)


class VoxelGridToQuadMesh(object):
    r""" Converts passed voxel to quad mesh

    Args:
        threshold (float): Threshold from which to make voxel binary.
        normalize (bool): Whether to scale the array to (-.5,.5).
    """
    def __init__(self, threshold: float, normalize: bool):
        self.thresh = threshold
        self.normalize = normalize

    def __call__(self, voxel: Type[VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel grid.

        Returns:
            (QuadMesh): Converted triangle mesh.
        """
        verts, faces = cvt.voxelgrid_to_quadmesh(voxel, self.thresh, self.normalize)
        return QuadMesh.from_tensors(vertices=verts, faces=faces)

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0}, normalize={1})'.\
            format(self.thresh, self.normalize)


class VoxelGridToPointCloud(object):
    r""" Converts  passed voxel to a pointcloud

    Args:
        num_points (int): Number of points in converted point cloud.
        thresh (float): Threshold from which to make voxel binary.
        mode (str):
            -'full': Sample the whole voxel model.
            -'surface': Sample only the surface voxels.
        normalize (bool): Whether to scale the array to (-.5,.5).
    """

    def __init__(self, num_points: int, threshold: float, mode: str, normalize: bool):
        self.num_points
        self.thresh = threshold
        self.mode
        self.normalize = normalize

    def __call__(self, voxel: Type[VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel grid.

        Returns:
            (torch.Tensor): Converted point cloud.
        """
        return cvt.voxelgrid_to_pointcloud(voxel, self.num_points, self.thresh, self.mode, self.normalize)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_points={0}'.format(self.num_points)
        format_string += ', threshold={0}'.format(self.threshold)
        format_string += ', mode={0}'.format(self.mode)
        format_string += ', normalize={0}'.format(self.normalize)
        return format_string


class VoxelGridToSDF(object):
    r""" Converts passed voxel to a signed distance fucntion.

    Args:
        voxel (torch.Tensor): Voxel grid
        thresh (float): threshold from which to make voxel binary
        normalize (bool): whether to scale the array to (0,1)

    Returns:
        a signed distance fucntion
    """
    def __init__(self, threshold: float, normalize: bool):
        self.thresh = threshold
        self.normalize = normalize

    def __call__(self, voxel: Type[VoxelGrid]):
        """
        Args:
            voxel (torch.Tensor): Voxel grid.

        Returns:
            (SDF): A signed distance function.
        """
        return cvt.voxelgrid_to_sdf(voxel, self.thresh, self.normalize)

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0}, normalize={1})'.\
            format(self.thresh, self.normalize)
