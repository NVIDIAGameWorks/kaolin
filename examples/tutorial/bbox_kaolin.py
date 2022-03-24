import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import kaolin


def vector_normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(p=2, dim=-1, keepdim=True)


def quaternion_to_matrix33(quat: torch.Tensor) -> torch.Tensor:
    q = vector_normalize(quat)

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    sqw = qw ** 2
    sqx = qx ** 2
    sqy = qy ** 2
    sqz = qz ** 2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs
    r0 = torch.stack([m00, m01, m02])
    r1 = torch.stack([m10, m11, m12])
    r2 = torch.stack([m20, m21, m22])
    mat33 = torch.stack([r0, r1, r2]).T
    return mat33


class DifferentiableBBox(torch.nn.Module):
    """Represents a differentiable 3d bounding box.

    Box is paramterized in terms of a fixed mesh and a learned (optimized)
    transformation in terms of rotation, scaling, and translation.
    """

    def __init__(self, mesh_file: Path):
        """Construct a 3d bounding from a mesh filepath.

        Args:
            mesh_file (str): Filepath to load the mesh. OBJ format.
        """
        super().__init__()

        assert mesh_file.exists(), f"File missing: {mesh_file.absolute()}."

        self._mesh = kaolin.io.obj.import_mesh(mesh_file, with_materials=True)
        self._vertices = None
        self._faces = None
        self._centers = torch.zeros((3,), dtype=torch.float, device="cuda", requires_grad=True)
        self._scales = torch.ones((3,), dtype=torch.float, device="cuda", requires_grad=True)
        # rotations as quaternion
        self._rotations = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda", requires_grad=True)
        self._preprocess()

    def _preprocess(self):
        self._vertices = self._mesh.vertices.cuda().unsqueeze(0) * 75
        self._vertices.requires_grad = False
        self._faces = self._mesh.faces.cuda()

    @property
    def vertices(self):
        rot = quaternion_to_matrix33(self._rotations)
        # scale, rotate, and translate vertices by optimized transform coordinates
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
    bbox: DifferentiableBBox,
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

    image_features, soft_mask, face_idx = kaolin.render.mesh.dibr_rasterization(
        image_shape[0],
        image_shape[1],
        face_vertices_camera[:, :, :, -1],
        face_vertices_image,
        face_attributes,
        face_normals[:, :, -1],
    )
    return soft_mask  # only aligning images by 2d silhouette


def draw_image(gt_mask, pred_mask):
    # mask shape is [w,h]
    canvas = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
    canvas[..., 2] = pred_mask.cpu().detach().numpy()
    canvas[..., 1] = gt_mask.cpu().detach().numpy()

    return np.clip(canvas, 0.0, 1.0)


def show_renders(dataset: torch.Tensor):
    with torch.no_grad():
        images = []
        for sample in dataset:
            gt_mask = sample["semantic"].cuda()
            camera_transform = sample["metadata"]["cam_transform"].cuda()
            camera_projection = sample["metadata"]["cam_proj"].cuda()

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

        return images


# hyperparameters
batch_size_hyper = 2
mask_weight_hyper = 1.0
mask_occupancy_hyper = 0.05
mask_overlap_hyper = 1.0
lr = 5e-2
scheduler_step_size = 5
scheduler_gamma = 0.5
num_epoch = 30

# dataset parameters
rendered_path = Path("../samples/rendered_clock/")
mesh_path = Path("../samples/bbox.obj")

# set up 2d image dataset including camera extrinsics:
num_views = len(glob.glob(os.path.join(rendered_path, "*_rgb.png")))
train_data = []
for i in range(num_views):
    data = kaolin.io.render.import_synthetic_view(rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)
dataloader = DataLoader(train_data, batch_size=batch_size_hyper, shuffle=True, pin_memory=True)

# set up model & optimization parameters
bbox = DifferentiableBBox(mesh_path)  # simple 3d box mesh data
optim = torch.optim.Adam(params=[bbox._centers, bbox._scales, bbox._rotations], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=scheduler_gamma)

# run training
image_list = []
for epoch in range(num_epoch):
    for idx, data in enumerate(dataloader):
        optim.zero_grad()

        # get image mask and camera extrinsics
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
        #   mask_occupancy = penalize larger masks
        #   mask_overlap = reward more overlap
        loss = mask_occupancy * mask_occupancy_hyper + mask_overlap * mask_overlap_hyper

        loss.backward()
        optim.step()

        # view training progress
        if idx % 10 == 0:
            test_batch_ids = [2, 5, 10]  # pick canonical test render views
            test_viz = [train_data[idx] for idx in test_batch_ids]
            # only keep 1 in 10 renders to reduce animation processing time
            image_list.append(show_renders(test_viz))

            cv2.imshow("test", image_list[-1][0])
            # minimal delay to ensure window updates
            # disable to remove thread-switching cost.
            k = cv2.waitKey(1)

    scheduler.step()
    print(f"loss on epoch {epoch:<}: {loss}")


# generate final animation
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
