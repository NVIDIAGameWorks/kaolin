import glob
import os
from typing import Tuple

import cv2
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pyrr import Quaternion, matrix44
from torch.utils.data import DataLoader

import kaolin

# hyperparameters
batch_size_hyper = 2
mask_weight_hyper = 1.0
mask_occupancy_hyper = 0.05
mask_overlap_hyper = 1.0
lr = 5e-2
scheduler_step_size = 5
scheduler_gamma = 0.5
num_epoch = 30
# TODO: stepsize scheduler

# load image(s) with camera extrinsics
rendered_path = "../samples/rendered_clock/"
num_views = len(glob.glob(os.path.join(rendered_path, "*_rgb.png")))
train_data = []
for i in range(num_views):
    data = kaolin.io.render.import_synthetic_view(rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)
dataloader = DataLoader(train_data, batch_size=batch_size_hyper, shuffle=True, pin_memory=True)


# modified from pyrr to use torch Tensor
def _euler_to_matrix33(pitch, roll, yaw, dtype=torch.float) -> torch.Tensor:

    sP = torch.sin(pitch)
    cP = torch.cos(pitch)
    sR = torch.sin(roll)
    cR = torch.cos(roll)
    sY = torch.sin(yaw)
    cY = torch.cos(yaw)

    m1 = torch.stack([cY * cP, -cY * sP * cR + sY * sR, cY * sP * sR + sY * cR])
    m2 = torch.stack([sP, cP * cR, -cP * sR])
    m3 = torch.stack([-sY * cP, sY * sP * cR + cY * sR, -sY * sP * sR + cY * cR])
    mat33 = torch.stack([m1, m2, m3]).cuda()
    return mat33


# set up mesh to optimize
class DiffBBox(torch.nn.Module):
    """Represents a 3d bounding box to fit."""

    def __init__(self, mesh_file: str):
        super().__init__()

        self._mesh = kaolin.io.obj.import_mesh(mesh_file, with_materials=True)
        self._vertices = None
        self._faces = None
        self._centers = torch.zeros((3,), dtype=torch.float, device="cuda", requires_grad=True)
        self._scales = torch.ones((3,), dtype=torch.float, device="cuda", requires_grad=True)
        # rotations as euler angles
        # self._rotations = torch.zeros((3,), dtype=torch.float, device="cuda", requires_grad=True)
        euler = np.random.uniform(0.0, 1.0, size=[3])
        self._rot_pitch = torch.nn.Parameter(torch.tensor(euler[0], dtype=torch.float))
        self._rot_roll = torch.nn.Parameter(torch.tensor(euler[1], dtype=torch.float))
        self._rot_yaw = torch.nn.Parameter(torch.tensor(euler[2], dtype=torch.float))
        self._preprocess()
        # TODO: add pose as quaternion. rotate verts

    def _preprocess(self):
        self._vertices = self._mesh.vertices.cuda().unsqueeze(0) * 75
        self._vertices.requires_grad = False
        self._faces = self._mesh.faces.cuda()

    def _get_transform(self):
        rot = None  # as 4x4
        scale = None  # as 4x4
        translation = None  # as 4x4
        return torch.matmul(translation, torch.matmul(rot, scale))

    @property
    def vertices(self):
        # rotate, scale in xyz, translate
        # euler = np.random.uniform(0.0, 1.0, size=[3])
        # rot = _euler_to_matrix33(euler)
        rot = _euler_to_matrix33(self._rot_pitch, self._rot_roll, self._rot_yaw)
        return (torch.matmul(self._vertices, rot) * self._scales) + self._centers

    @property
    def faces(self):
        return self._faces


def overlap(lhs_mask, rhs_mask):
    batch_size, height, width = lhs_mask.shape
    assert rhs_mask.shape == lhs_mask.shape
    sil_mul = lhs_mask * rhs_mask
    sil_area = torch.sum(sil_mul.reshape(batch_size, -1), dim=1)

    return 1 - torch.mean(sil_area / (height * width))


def occupancy(mask):
    batch_size, height, width = mask.shape
    mask_area = torch.sum(mask.reshape(batch_size, -1), dim=1)
    return torch.mean(mask_area / (height * width))


def project_to_2d(
    bbox: DiffBBox,
    batch_size: int,
    image_shape: Tuple[int, int],
    camera_transform,
    camera_projection,
) -> torch.Tensor:
    (face_vertices_camera, face_vertices_image, face_normals,) = kaolin.render.mesh.prepare_vertices(
        bbox.vertices.repeat(batch_size, 1, 1),
        bbox.faces,
        camera_projection,
        camera_transform=camera_transform,
    )

    nb_faces = bbox.faces.shape[0]
    face_attributes = [torch.ones((batch_size, nb_faces, 3, 1), device="cuda")]

    # TODO: not sure what this is doing / expecting
    image_features, soft_mask, face_idx = kaolin.render.mesh.dibr_rasterization(
        image_shape[0],
        image_shape[1],
        face_vertices_camera[:, :, :, -1],
        face_vertices_image,
        face_attributes,
        face_normals[:, :, -1],
    )
    return soft_mask


def draw_image(gt_mask, pred_mask):
    # assumption: mask shape is [b,w,h]
    canvas = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
    canvas[..., 2] = pred_mask.cpu().detach().numpy()
    canvas[..., 1] = gt_mask.cpu().detach().numpy()

    return np.clip(canvas, 0.0, 1.0)


def show_renders(dataset: torch.Tensor):
    with torch.no_grad():
        batch_size = len(dataset)
        images = []
        for idx, data in enumerate(dataset):
            gt_mask = data["semantic"].cuda()
            camera_transform = data["metadata"]["cam_transform"].cuda()
            camera_projection = data["metadata"]["cam_proj"].cuda()

            # project model mesh onto 2d image
            img_shape = (gt_mask.shape[0], gt_mask.shape[1])
            soft_mask = project_to_2d(
                bbox,
                batch_size=1,
                image_shape=img_shape,
                camera_transform=camera_transform,
                camera_projection=camera_projection,
            )
            image = draw_image(gt_mask.squeeze(), soft_mask)
            images.append(image)

        # fig, ax = plt.subplots(1, batch_size, figsize=(4, 6))
        # for i in range(batch_size):
        #     ax[i].imshow(images[i])

        return images


# set up model & optimization
bbox = DiffBBox("../samples/bbox.obj")
optim = torch.optim.Adam(params=[bbox._centers, bbox._scales, *bbox.parameters()], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=scheduler_gamma)


image_list = []
for epoch in range(num_epoch):
    for idx, data in enumerate(dataloader):
        optim.zero_grad()

        # get image mask and camera properties
        gt_mask = data["semantic"].cuda()
        camera_transform = data["metadata"]["cam_transform"].cuda()
        camera_projection = data["metadata"]["cam_proj"].cuda()

        # project model mesh onto 2d image
        img_shape = (gt_mask.shape[1], gt_mask.shape[2])
        soft_mask = project_to_2d(
            bbox,
            batch_size=batch_size_hyper,
            image_shape=img_shape,
            camera_transform=camera_transform,
            camera_projection=camera_projection,
        )

        # compute overlap of projection with ground truth segmentation mask
        mask_loss: torch.Tensor = kaolin.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))
        mask_occupancy = occupancy(soft_mask)
        mask_overlap = overlap(soft_mask, gt_mask.squeeze(-1))

        # loss = mask_loss * mask_weight
        # penalize larger masks
        # reward more overlap
        loss = mask_occupancy * mask_occupancy_hyper + mask_overlap * mask_overlap_hyper

        loss.backward()
        optim.step()

        if idx % 10 == 0:
            test_batch_ids = [2, 5, 10]
            test_viz = [train_data[idx] for idx in test_batch_ids]
            image_list.append(show_renders(test_viz))

            cv2.imshow("test", image_list[-1][0])
            k = cv2.waitKey(1)

    scheduler.step()
    print(f"{epoch} loss: {loss}")


# generate final plot
num_subplots = len(test_batch_ids)
f, ax = plt.subplots(ncols=num_subplots)
ims = []
for i in range(len(image_list)):
    sp_ims = []
    for j in range(num_subplots):
        im = ax[j].imshow(image_list[i][j], animated=True)
        sp_ims.append(im)
    # show first to not have blinking animation
    if i == 0:
        for j in range(num_subplots):
            ax[j].imshow(image_list[i][j])
    ims.append(sp_ims)
ani = animation.ArtistAnimation(f, ims, interval=60, blit=True, repeat_delay=200)
ani.save("animation.gif")
