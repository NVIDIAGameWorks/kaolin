from PIL import Image
import torch
import cv2
import numpy as np
import kaolin
from matplotlib import pyplot as plt

from typing import Tuple
from torch.utils.data import DataLoader

import os
import glob

# hyperparameters
batch_size_hyper = 2
mask_weight = 1.0
lr = 5e-2
num_epoch = 300

# load image(s) with camera extrinsics
rendered_path = "../samples/rendered_clock/"
num_views = len(glob.glob(os.path.join(rendered_path,'*_rgb.png')))
train_data = []
for i in range(num_views):
    data = kaolin.io.render.import_synthetic_view(rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)
dataloader = DataLoader(train_data, batch_size=batch_size_hyper, shuffle=True, pin_memory=True)

# set up mesh to optimize
class DiffBBox:
    """Represents a 3d bounding box to fit."""
    def __init__(self, mesh_file: str):
        self.mesh = kaolin.io.obj.import_mesh(mesh_file, with_materials=True)
        self.vertices = None
        self.faces = None
        self._preprocess()
        # TODO: reparameterize as center and scale - ie NOT change vertices themselves
    
    def _preprocess(self):
        self.vertices = self.mesh.vertices.cuda().unsqueeze(0) * 75
        self.vertices.requires_grad = True
        self.faces = self.mesh.faces.cuda()

def project_to_2d(bbox: DiffBBox, batch_size: int, image_shape: Tuple[int, int], camera_transform, camera_projection) -> torch.Tensor:
    face_vertices_camera, face_vertices_image, face_normals = kaolin.render.mesh.prepare_vertices(
                bbox.vertices.repeat(batch_size, 1, 1),
                bbox.faces, camera_projection,camera_transform=camera_transform
            )

    nb_faces = bbox.faces.shape[0]
    face_attributes = [
        torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
    ]

    # TODO: not sure what this is doing / expecting
    image_features, soft_mask, face_idx = kaolin.render.mesh.dibr_rasterization(image_shape[0], image_shape[1], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes, face_normals[:, :, -1])
    return soft_mask


def draw_image(gt_mask, pred_mask):
    # assumption: mask shape is [b,w,h]
    canvas = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
    canvas[...,2] = pred_mask.cpu().detach().numpy()
    canvas[...,1] = gt_mask.cpu().detach().numpy()

    return canvas*255

import matplotlib.animation as animation

def show_renders(dataset: torch.Tensor):
    with torch.no_grad():
        batch_size = len(dataset)
        images = []
        for idx, data in enumerate(dataset):
            gt_mask = data['semantic'].cuda()
            camera_transform = data['metadata']['cam_transform'].cuda()
            camera_projection = data['metadata']['cam_proj'].cuda()
            
            # project model mesh onto 2d image
            img_shape = (gt_mask.shape[0], gt_mask.shape[1])
            soft_mask = project_to_2d(bbox, batch_size=1, image_shape=img_shape, camera_transform=camera_transform, camera_projection=camera_projection)
            image = draw_image(gt_mask.squeeze(), soft_mask)
            images.append(image)

        # fig, ax = plt.subplots(1, batch_size, figsize=(4, 6))
        # for i in range(batch_size):
        #     ax[i].imshow(images[i])
        
        return images

# set up model & optimization
bbox = DiffBBox('../samples/bbox.obj')
optim = torch.optim.Adam(params=[bbox.vertices], lr=lr)

image_list = []
for epoch in range(num_epoch):
    for idx, data in enumerate(dataloader):
        optim.zero_grad()

        # get image mask and camera properties
        gt_mask = data['semantic'].cuda()
        camera_transform = data['metadata']['cam_transform'].cuda()
        camera_projection = data['metadata']['cam_proj'].cuda()
        
        # project model mesh onto 2d image
        img_shape = (gt_mask.shape[1], gt_mask.shape[2])
        soft_mask = project_to_2d(bbox, batch_size=batch_size_hyper, image_shape=img_shape, camera_transform=camera_transform, camera_projection=camera_projection)

        # compute overlap of projection with ground truth segmentation mask
        mask_loss: torch.Tensor = kaolin.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))

        loss = mask_loss * mask_weight

        loss.backward()
        optim.step()

    print(f"{epoch} loss: {loss}")
    test_batch_ids = [2, 5, 10]
    test_viz = [train_data[idx] for idx in test_batch_ids]
    image_list.append(show_renders(test_viz))


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
ani = animation.ArtistAnimation(f, ims, interval=30, blit=True, repeat_delay=200)
ani.save("animation.gif")
# plt.show()
