from PIL import Image
import torch
import numpy as np
import kaolin
from matplotlib import pyplot as plt

from typing import Tuple
from torch.utils.data import DataLoader

import os
import glob

# hyperparameters
batch_size = 2
mask_weight = 1

# load image(s) with camera extrinsics
rendered_path = "../samples/rendered_clock/"
num_views = len(glob.glob(os.path.join(rendered_path,'*_rgb.png')))
train_data = []
for i in range(num_views):
    data = kaolin.io.render.import_synthetic_view(rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

# set up model parameters for hull
#   TODO: differentiable parameters
class DiffBBox:
    def __init__(self, mesh_file: str):
        self.mesh = kaolin.io.obj.import_mesh(mesh_file, with_materials=True)
        self.vertices = None
        self.faces = None
        self._preprocess()
    
    def _preprocess(self):
        self.vertices = self.mesh.vertices.cuda().unsqueeze(0) * 75
        self.vertices.requires_grad = True
        self.faces = self.mesh.faces.cuda()

def project_to_2d(bbox: DiffBBox, image_shape: Tuple[int, int], camera_transform, camera_projection):
    face_vertices_camera, face_vertices_image, face_normals = kaolin.render.mesh.prepare_vertices(
                bbox.vertices.repeat(batch_size, 1, 1),
                bbox.faces, camera_projection,camera_transform=camera_transform
            )

    nb_faces = bbox.faces.shape[0]
    face_attributes = [
        torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
    ]

    image_features, soft_mask, face_idx = kaolin.render.mesh.dibr_rasterization(image_shape[0], image_shape[1], face_vertices_camera[:, :, :, -1], face_vertices_image, face_attributes, face_normals[:, :, -1])
    return soft_mask

bbox = DiffBBox('../samples/bbox.obj')

for idx, data in enumerate(dataloader):
    # get image mask and camera properties
    gt_mask = data['semantic'].cuda()
    camera_transform = data['metadata']['cam_transform'].cuda()
    camera_projection = data['metadata']['cam_proj'].cuda()
    
    # project model mesh onto 2d image
    img_shape = (gt_mask.shape[1], gt_mask.shape[2])
    soft_mask = project_to_2d(bbox, image_shape=img_shape, camera_transform=camera_transform, camera_projection=camera_projection)

    # compute overlap of projection with ground truth segmentation mask
    mask_loss = kaolin.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))

    loss = mask_loss * mask_weight
    print(f"loss: {loss}")
