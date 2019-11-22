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

import sys
import os
from pathlib import Path
import torch
import torch.utils.data as data
import warnings
import urllib.request
import zipfile
import json
import re
from collections import OrderedDict
from glob import glob
import numpy as np
import random

from tqdm import tqdm
import scipy.sparse
import tarfile
from PIL import Image

import kaolin as kal
from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.rep.QuadMesh import QuadMesh

from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import meshfunc
from kaolin.transforms import voxelfunc
from kaolin.transforms import transforms as tfs
from kaolin import helpers
import kaolin.conversions.meshconversions as mesh_cvt


# Synset to Label mapping (for ShapeNet core classes)
synset_to_label = {'04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
                   '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
                   '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
                   '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
                   '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
                   '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
                   '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
                   '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
                   '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
                   '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
                   '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}


class print_wrapper(object):
    def __init__(self, text, logger=sys.stdout.write):
        self.text = text
        self.logger = logger

    def __enter__(self):
        self.logger(self.text)

    def __exit__(self, *args):
        self.logger("\t[done]\n")


def tqdm_hook(t, timeout=1):
    """Taken from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py

    Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.request.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_shapenet_class(syn: str, shapenet_location: str, download: bool):
    r"""Downloads a shapenet to a specified directory.

    You may complete this function to automate downloading from a
    central source.
    """
    NotImplemented


def download_images(shapenet_root: str):
    r"""Downloads shapenet core class images to a specified directory

    You may complete this function to automate downloading from a
    central source.
    """
    NotImplemented


def _convert_categories(categories):
    assert categories is not None, 'List of categories cannot be empty!'
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            ShapeNetCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets


class ShapeNet_Meshes(data.Dataset):
    r"""ShapeNet Dataset class for meshes.

    Args:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        download (bool): download the shapenet class if not found
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, path: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
        }

    Example:
        >>> meshes = ShapeNet_Meshes(root='./datasets/ShapeNet/', cache_dir='cache', download=True)
        >>> obj = next(iter(meshes))
        >>> obj['data']['vertices'].shape
        torch.Size([2133, 3])
        >>> obj['data']['faces'].shape
        torch.Size([1910, 3])
    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 download: bool = False, split: float = .7, no_progress: bool = False):
        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                download_shapenet(syn, str(self.root), download)

            # find all objects in the class
            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.paths)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        synset_idx = self.synset_idxs[index]
        obj_location = self.paths[index] / 'model.obj'
        mesh = TriangleMesh.from_obj(str(obj_location))

        data['vertices'] = mesh.vertices
        data['faces'] = mesh.faces
        attributes['name'] = self.names[index]
        attributes['path'] = obj_location
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Images(data.Dataset):
    r"""ShapeNet Dataset class for images.

    Arguments:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): if true use the training set, else use the test set
        split (float): amount of dataset that is training out of
        download (bool): if true download dataset set not found
        views (int): number of viewpoints per object to load
        transform (torchvision.transforms) : transformation to apply to images
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, path: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
            params: {
                cam_mat: torch.Tensor,
                cam_pos: torch.Tensor,
                azi: float,
                elevation: float,
                distance: float
            }
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> images = ShapeNet_Images(root ='./datasets/', download = True)
        >>> train_loader = DataLoader(images, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> image = obj['data']['imgs']
        >>> image.shape
        torch.Size([10, 4, 137, 137])
    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, download: bool = True, views: int = 23, transform=None,
                 no_progress: bool = False):
        self.root = root
        self.synsets = _convert_categories(categories)
        self.transform = transform
        self.views = views
        self.names = []
        self.synset_idx = []

        self.shapenet_root = Path(root) / 'ShapeNet'
        shapenet_img_root = Path(self.shapenet_root) / 'images'
        # check if images already exists and if not download them
        if not shapenet_img_root.exists():
            shapenet_img_root.mkdir()
            assert download, f'ShapeNet Images are not found, and download is set to False'
            download_images(str(self.shapenet_root))

        # find all needed images
        for i in tqdm(range(len(self.synsets)), disable=no_progress):
            syn = self.synsets[i]
            class_target = shapenet_img_root / syn
            assert class_target.exists(), \
                "ShapeNet class, {0}, is not found".format(syn)

            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.names += models

            self.synset_idx += [i] * len(models)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        view_num = random.randrange(0, self.views)
        # load and process image
        img = Image.open(str(img_name / f'rendering/{view_num:02}.png'))
        # apply transformations
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(np.array(img))
            img = img.permute(2, 1, 0)
            img = img / 255.
        # load and process camera parameters
        param_location = img_name / 'rendering/rendering_metadata.txt'
        azimuth, elevation, _, distance, _ = np.loadtxt(param_location)[view_num]
        cam_params = kal.math.geometry.transformations.compute_camera_params(
            azimuth, elevation, distance)

        data['images'] = img
        data['params'] = dict()
        data['params']['cam_mat'] = cam_params[0]
        data['params']['cam_pos'] = cam_params[1]
        data['params']['azi'] = azimuth
        data['params']['elevation'] = elevation
        data['params']['distance'] = distance
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.synsets[self.synset_idx[index]]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Voxels(data.Dataset):
    r"""ShapeNet Dataset class for voxels.

    Args:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        download (bool): download the shapenet class if not found
        resolutions (list): list of resolutions to be returned
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {[res]: torch.Tensor}
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> voxels = ShapeNet_Voxels(root='../data/', download=True)
        >>> train_loader = DataLoader(voxels, batch_size=10, shuffle=True, num_workers=8 )
        >>> obj = next(iter(train_loader))
        >>> obj['data']['128'].shape
        torch.Size([10, 128, 128, 128])

    """
    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 download: bool = True, split: float = .7, resolutions=[128, 32],
                 no_progress: bool = False):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'voxels'
        self.cache_transforms = {}
        self.params = {
            'resolutions': resolutions,
        }
        mesh_dataset = ShapeNet_Meshes(root=root,
                                       cache_dir=cache_dir,
                                       categories=categories,
                                       train=train,
                                       download=download,
                                       split=split,
                                       no_progress=no_progress)
        self.names = mesh_dataset.names
        self.synset_idxs = mesh_dataset.synset_idxs
        self.synsets = mesh_dataset.synsets
        self.labels = mesh_dataset.labels

        for res in self.params['resolutions']:
            self.cache_transforms[res] = tfs.CacheCompose([
                tfs.TriangleMeshToVoxelGrid(res, normalize=False, vertex_offset=0.5),
                tfs.FillVoxelGrid(thresh=0.5),
                tfs.ExtractProjectOdmsFromVoxelGrid()
            ], self.cache_dir)

            desc = 'converting to voxels'
            for idx in tqdm(range(len(mesh_dataset)), desc=desc, disable=no_progress):
                name = mesh_dataset.names[idx]
                if name not in self.cache_transforms[res].cached_ids:
                    sample = mesh_dataset[idx]
                    mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                     sample['data']['faces'])
                    self.cache_transforms[res](name, mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        for res in self.params['resolutions']:
            data[str(res)] = self.cache_transforms[res](name)
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Surface_Meshes(data.Dataset):
    r"""ShapeNet Dataset class for watertight meshes with only the surface preserved.

    Arguments:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        download (bool): download the shapenet class if not found
        resolution (int): resolution of voxel object to use when converting
        smoothing_iteration (int): number of applications of laplacian smoothing
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
        }

    Example:
        >>> surface_meshes = ShapeNet_Surface_Meshes(root ='../data/', download = True)
        >>> obj = next(iter(surface_meshes))
        >>> obj['data']['vertices'].shape
        torch.Size([11617, 3])
        >>> obj['data']['faces'].shape
        torch.Size([23246, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 download: bool = True, split: float = .7, resolution: int = 100,
                 smoothing_iterations: int = 3, mode='Tri', no_progress: bool = False):
        assert mode in ['Tri', 'Quad']

        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'surface_meshes'
        dataset_params = {
            'root': root,
            'cache_dir': cache_dir,
            'categories': categories,
            'train': train,
            'download': download,
            'split': split,
            'no_progress': no_progress,
        }
        self.params = {
            'resolution': resolution,
            'smoothing_iterations': smoothing_iterations,
            'mode': mode,
        }

        mesh_dataset = ShapeNet_Meshes(**dataset_params)
        voxel_dataset = ShapeNet_Voxels(**dataset_params, resolutions=[resolution])
        combined_dataset = ShapeNet_Combination([mesh_dataset, voxel_dataset])

        self.names = combined_dataset.names
        self.synset_idxs = combined_dataset.synset_idxs
        self.synsets = combined_dataset.synsets
        self.labels = combined_dataset.labels

        if mode == 'Tri':
            mesh_conversion = tfs.VoxelGridToTriangleMesh(threshold=0.5,
                                                          mode='marching_cubes',
                                                          normalize=False)
        else:
            mesh_conversion = tfs.VoxelGridToQuadMesh(threshold=0.5,
                                                      normalize=False)

        def convert(og_mesh, voxel):
            transforms = tfs.Compose([mesh_conversion,
                                     tfs.MeshLaplacianSmoothing(smoothing_iterations)])

            new_mesh = transforms(voxel)
            new_mesh.vertices = pcfunc.realign(new_mesh.vertices, og_mesh.vertices)
            return {'vertices': new_mesh.vertices, 'faces': new_mesh.faces}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to surface meshes'
        for idx in tqdm(range(len(combined_dataset)), desc=desc, disable=no_progress):
            name = combined_dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                sample = combined_dataset[idx]
                voxel = sample['data'][str(resolution)]
                og_mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                    sample['data']['faces'])
                self.cache_convert(name, og_mesh=og_mesh, voxel=voxel)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        data = self.cache_convert(name)
        mesh = TriangleMesh.from_tensors(data['vertices'], data['faces'])
        data['adj'] = mesh.compute_adjacency_matrix_sparse().coalesce()
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Points(data.Dataset):
    r"""ShapeNet Dataset class for sampled point cloud from each object.

    Args:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        download (bool): download the shapenet class if not found
        num_points (int): number of point sampled on mesh
        smoothing_iteration (int): number of application of laplacian smoothing
        surface (bool): if only the surface of the original mesh should be used
        resolution (int): resolution of voxel object to use when converting
        normals (bool): should the normals of the points be saved
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, synset: str, label: str},
            data: {points: torch.Tensor, normals: torch.Tensor}
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> points = ShapeNet_Points(root ='../data/', download = True)
        >>> train_loader = DataLoader(points, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['points'].shape
        torch.Size([10, 5000, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 download: bool = True, split: float = .7, num_points: int = 5000, smoothing_iterations=3,
                 surface=True, resolution=100, normals=True, no_progress: bool = False):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'points'

        dataset_params = {
            'root': root,
            'cache_dir': cache_dir,
            'categories': categories,
            'train': train,
            'download': download,
            'split': split,
            'no_progress': no_progress,
        }
        self.params = {
            'num_points': num_points,
            'smoothing_iterations': smoothing_iterations,
            'surface': surface,
            'resolution': resolution,
            'normals': normals,
        }

        if surface:
            dataset = ShapeNet_Surface_Meshes(**dataset_params,
                                              resolution=resolution,
                                              smoothing_iterations=smoothing_iterations)
        else:
            dataset = ShapeNet_Meshes(**dataset_params)

        self.names = dataset.names
        self.synset_idxs = dataset.synset_idxs
        self.synsets = dataset.synsets
        self.labels = dataset.labels

        def convert(mesh):
            points, face_choices = mesh_cvt.trianglemesh_to_pointcloud(mesh, num_points)
            face_normals = mesh.compute_face_normals()
            point_normals = face_normals[face_choices]
            return {'points': points, 'normals': point_normals}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to points'
        for idx in tqdm(range(len(dataset)), desc=desc, disable=no_progress):
            name = dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                idx = dataset.names.index(name)
                sample = dataset[idx]
                mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                 sample['data']['faces'])
                self.cache_convert(name, mesh=mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        data = self.cache_convert(name)
        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_SDF_Points(data.Dataset):
    r"""ShapeNet Dataset class for signed distance functions.

    Args:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): return the training set else the test set
        split (float): amount of dataset that is training out of 1
        download (bool): download the shapenet class if not found
        resolution (int): resolution of voxel object to use when converting
        num_points (int): number of sdf points sampled on mesh
        occ (bool): should only occupancy values be returned instead of distances
        smoothing_iteration (int): number of application of laplacian smoothing
        sample_box (bool): whether to sample only from within mesh extents
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

            dict: {
                attributes: {name: str, synset: str, label: str},
                data: {
                    Union['occ_values', 'sdf_distances']: torch.Tensor,
                    Union['occ_points, 'sdf_points']: torch.Tensor}
            }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> sdf_points = ShapeNet_SDF_Points(root ='../data/', download = True)
        >>> train_loader = DataLoader(sdf_points, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['sdf_points'].shape
        torch.Size([10, 5000, 3])

    """

    def __init__(self, root: str, cache_dir: str, categories: list = ['chair'], train: bool = True,
                 download: bool = True, split: float = .7, resolution: int = 100, num_points: int = 5000,
                 occ: bool = False, smoothing_iterations: int = 3, sample_box=True, no_progress: bool = False):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) / 'sdf_points'

        self.params = {
            'resolution': resolution,
            'num_points': num_points,
            'occ': occ,
            'smoothing_iterations': smoothing_iterations,
            'sample_box': sample_box,
        }

        surface_mesh_dataset = ShapeNet_Surface_Meshes(root=root,
                                                       cache_dir=cache_dir,
                                                       categories=categories,
                                                       train=train,
                                                       download=download,
                                                       split=split,
                                                       resolution=resolution,
                                                       smoothing_iterations=smoothing_iterations,
                                                       no_progress=no_progress)

        self.names = surface_mesh_dataset.names
        self.synset_idxs = surface_mesh_dataset.synset_idxs
        self.synsets = surface_mesh_dataset.synsets
        self.labels = surface_mesh_dataset.labels

        def convert(mesh):
            sdf = mesh_cvt.trianglemesh_to_sdf(mesh, num_points)
            bbox_true = torch.stack((mesh.vertices.min(dim=0)[0],
                                     mesh.vertices.max(dim=0)[0]), dim=1).view(-1)
            points = 1.05 * (torch.rand(self.params['num_points'], 3).to(mesh.vertices.device) - .5)
            distances = sdf(points)
            return {'points': points, 'distances': distances, 'bbox': bbox_true}

        self.cache_convert = helpers.Cache(convert, self.cache_dir,
                                           cache_key=helpers._get_hash(self.params))

        desc = 'converting to sdf points'
        for idx in tqdm(range(len(surface_mesh_dataset)), desc=desc, disable=no_progress):
            name = surface_mesh_dataset.names[idx]
            if name not in self.cache_convert.cached_ids:
                idx = surface_mesh_dataset.names.index(name)
                sample = surface_mesh_dataset[idx]
                mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                 sample['data']['faces'])

                # Use cuda if available to speed up conversion
                if torch.cuda.is_available():
                    mesh.cuda()
                self.cache_convert(name, mesh=mesh)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        cached_data = self.cache_convert(name)
        points = cached_data['points']
        distances = cached_data['distances']

        if self.params['sample_box']:
            bbox_values = kal.rep.bounding_points(points, cached_data['bbox'])
            points = points[bbox_values]
            distances = distances[bbox_values]

        selection = np.random.randint(points.shape[0], size=self.params['num_points'])

        if self.params['occ']:
            data['occ_values'] = distances[selection] <= 0
            data['occ_points'] = points[selection]
        else:
            data['sdf_distances'] = distances[selection]
            data['sdf_points'] = points[selection]

        attributes['name'] = self.names[index]
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Tags(data.Dataset):
    r"""ShapeNet Dataset class for tags.

    Args:
        dataset (kal.dataloader.shapenet.ShapeNet): One of the ShapeNet datasets
        download (bool): If True will load taxonomy of objects if it is not loaded yet
        transform (...) : transformation to apply to tags

    Returns:
        dict: Dictionary with key for the input tags encod and : 'tag_enc':

    Example:
        >>> from torch.utils.data import DataLoader
        >>> meshes = ShapeNet_Meshes(root='../data/', download=True)
        >>> tags = ShapeNet_Tags(meshes)
        >>> train_loader = DataLoader(tags, batch_size=10, shuffle=True, num_workers=8 )
        >>> obj = next(iter(train_loader))
        >>> obj['data']['tag_enc'].shape
        torch.Size([10, N])

    """
    def __init__(self, dataset, download=True, tag_aug=True):
        self.root = dataset.root
        self.paths = dataset.paths
        self.synset_idxs = dataset.synset_idxs
        self.synsets = dataset.synsets
        self.names = dataset.names
        self.tag_aug = tag_aug

        # load taxonomy
        self.get_taxonomy()
        # get the mapping of sample indexes to file names
        self.get_name_index_mapping()

        self.all_tags = []
        self.name_tag_map = {}
        self.tag_name_map = OrderedDict()

        # get labels for instances
        for synset_index, synset in enumerate(self.synsets):
            tag = self.get_tags_from_taxonomy(synset)
            indexes = np.where(np.array(self.synset_idxs) == synset_index)[0]
            instances = [self.names[ind] for ind in indexes]

            for name in instances:
                # get tags from labels
                tag_list = self.get_tags_from_str(tag)
                self.all_tags.extend(tag_list)

                if name in self.name_to_index.keys():
                    # populate the mappings
                    self.update_name_tag_mappings(name, tag_list)
                else:
                    print(name)

        # compute unique tags
        self.all_tags = list(np.unique(self.all_tags))
        # get indexes of tags for one-hot encoding
        self.get_tag_indexes()

    def get_tag_indexes(self):
        self.tag_index = {}
        for index, tag in enumerate(self.tag_name_map.keys()):
            self.tag_index[tag] = index

    def get_taxonomy(self):
        r"""Download the taxonomy from the web."""
        taxonomy_location = os.path.join(self.root, 'taxonomy.json')
        if not os.path.exists(taxonomy_location):
            with print_wrapper("Downloading taxonomy ..."):
                taxonomy_web_location = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/taxonomy.json'
                urllib.request.urlretrieve(taxonomy_web_location,
                                           filename=taxonomy_location)

    def update_name_tag_mappings(self, name, tag_list):
        r"""Create a mapping from the file name to its label and from each tag to a list of files with this tag."""
        # populate tag to file name mapping
        self.name_tag_map[name] = tag_list
        # populate inverse mapping
        for tag in tag_list:
            if tag not in self.tag_name_map.keys():
                self.tag_name_map[tag] = [name]
            else:
                self.tag_name_map[tag].append(name)

    def get_name_index_mapping(self):
        r"""Calculate the mapping between file names and the sample indices"""
        self.name_to_index = {}
        for index, name in enumerate(self.names):
            self.name_to_index[name] = index

    def get_tags_from_str(self, tags_str, inverse_order=True, forbidden_symbols=[" ", "/", "-", "\*"]):
        r"""Process the tag string and return a list of tags. ``Note``: The tags that contain forbidden_symbols are ignored.

        Args:
            tags_str (str): string with comma separated tags.
            inverse_order (bool): reverse the order of tags

        Returns:
            list of tags.
        """
        output_list = [
            tag.strip() for tag in tags_str.split(',')
            if re.match(".*(:?{}).*".format("|".join(forbidden_symbols)), tag.strip()) is None
        ]
        if inverse_order:
            output_list = output_list[::-1]
        return output_list

    def get_category_paths(self, category='chair'):
        r"""Get the list of SynSet IDs and the respective tags based on the taxonomy.

        Args:
            category (str): category of the object that needs to be retrieved.

        Returns:
            synsetIds (list): list of synsets
            tags (list): list of tags for each synset
        """
        with open(os.path.join(self.root, 'taxonomy.json'), 'r') as json_f:
            taxonomy = json.load(json_f)

        synsetIds, children = [], []
        parent_tags, tags = [], []

        for c in taxonomy:
            tag = c['name']

            matchObj = True
            if category is not None:
                matchObj = re.search(
                    r'(?<![a-zA-Z0-9])' + category + '(?![a-zA-Z0-9])', tag,
                    re.M | re.I)

            if matchObj:
                sid = c['synsetId']
                if sid not in synsetIds:
                    synsetIds.append(sid)
                    tags.append(tag)
                for childId in c['children']:
                    if childId not in children:
                        children.append(childId)
                        parent_tags.append(tag)

        while len(children) > 0:
            new_children = []
            new_parent_tags = []
            for c in taxonomy:
                sid = c['synsetId']
                if sid in children and sid not in synsetIds:
                    synsetIds.append(sid)
                    i = children.index(sid)
                    tag = c['name'] + ',' + parent_tags[i]
                    tags.append(tag)
                    for childId in c['children']:
                        if childId not in new_children:
                            new_children.append(childId)
                            new_parent_tags.append(tag)

            children = new_children
            parent_tags = new_parent_tags

        return synsetIds, tags

    def get_tags_from_taxonomy(self, synset, verbose=False):
        r"""Load category based on the ShapeNet taxonomy.

        Args:
            category (str): catergory of the object that needs to be retrieved.
            verbose (bool): If ``True`` - print some additional information.

        Returns:
            instances (list): list of object paths that contain objects from the requested category.
            tags (list): list of tags for each instance.
        """
        category = synset_to_label[synset]
        synsetIds, tags = self.get_category_paths(category=category)
        sid = synsetIds.index(synset)
        tag = tags[sid]

        return tag

    def rand_drop_tag(self, tag_list):
        r"""Drop some tags from the label randomly and return the orderring number of the most specific tag."""
        if len(tag_list) == 1:
            return tag_list, 0
        else:
            tags_to_keep = np.random.randint(1, len(tag_list) + 1)
            res_tag_ind = np.random.choice(range(len(tag_list)),
                                           tags_to_keep,
                                           replace=False)
            max_ind = max(res_tag_ind)
            res_tag_list = [tag_list[el] for el in res_tag_ind]
            return res_tag_list, max_ind

    def tag_proc(self, tag_list):
        """Get the embedding from the list of tags. By default this function does
        one-hot encoding of the tags, but can be replaced by more complex encodings.

        Args:
            tag_list (list): List of textual tags that need to be encoded.
        """
        embed = np.zeros(len(self.tag_index), dtype=np.uint8)
        for tag in tag_list:
            embed[self.tag_index[tag]] = 1

        return torch.from_numpy(embed)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        synset_idx = self.synset_idxs[index]

        full_tag = self.name_tag_map[self.names[index]]

        # do tags augmentation
        if self.tag_aug:
            input_tags, last_tag_id = self.rand_drop_tag(full_tag)
        else:
            input_tags = full_tag
            last_tag_id = len(full_tag) - 1

        # tag encodings
        data['tag_inp'] = self.tag_proc(input_tags)
        data['tag_full'] = self.tag_proc(full_tag[:(last_tag_id + 1)])
        data['tag_label'] = self.tag_proc(full_tag)
        # length of tags per
        data['tag_inp_len'] = torch.tensor(len(input_tags))
        data['tag_full_len'] = torch.tensor(len(full_tag[:(last_tag_id + 1)]))
        data['tag_label_len'] = torch.tensor(len(full_tag))

        attributes['name'] = name
        attributes['synset'] = self.synsets[synset_idx]
        attributes['label'] = self.labels[synset_idx]
        return {'data': data, 'attributes': attributes}


class ShapeNet_Combination(data.Dataset):
    r"""ShapeNet Dataset class for combinations of representations.

    Arguments:
        dataset (list): List of datasets to be combined
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        root (str): Path to the root directory of the ShapeNet dataset.
        train (bool): if true use the training set, else use the test set
        download (bool): if true download dataset set not found

    Returns:
        dict: Dictionary with keys indicated by passed datasets

    Example:

        >>> from torch.utils.data import DataLoader
        >>> shapenet = ShapeNet_Meshes(root ='../data/', download = True)
        >>> voxels = ShapeNet_Voxels(shapenet)
        >>> images = ShapeNet_Images(shapenet)
        >>> points = ShapeNet_Points(shapenet)
        >>> dataset = ShapeNet_Combination([voxels, images, points])
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> for key in obj['data']:
        ...     print (key)
        ...
        params
        128
        32
        imgs
        cam_mat
        cam_pos
        azi
        elevation
        distance
        points
        normals
    """

    def __init__(self, datasets):
        self.names = datasets[0].names
        self.root = datasets[0].root
        self.synset_idxs = datasets[0].synset_idxs
        self.synsets = datasets[0].synsets
        self.labels = datasets[0].labels
        self.datasets = datasets

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        obj = self.datasets[0][index]

        for ds in self.datasets:
            obj['data'].update(ds[index]['data'])

        return obj
