import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def directional_lighting(light, normals, light_intensity=0.5, light_color=(1,1,1), 
                         light_direction=(0,1,0)):
    # normals: [nb, :, 3]

    device = light.device

    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = torch.tensor(light_color, dtype=torch.float32, device=device)
    elif isinstance(light_color, np.ndarray):
        light_color = torch.from_numpy(light_color).float().to(device)
    if isinstance(light_direction, tuple) or isinstance(light_direction, list):
        light_direction = torch.tensor(light_direction, dtype=torch.float32, device=device)
    elif isinstance(light_direction, np.ndarray):
        light_direction = torch.from_numpy(light_direction).float().to(device)
    if light_color.ndimension() == 1:
        light_color = light_color[None, :]
    if light_direction.ndimension() == 1:
        light_direction = light_direction[None, :] #[nb, 3]

    cosine = F.relu(torch.sum(normals * light_direction, dim=2)) #[]
    light += light_intensity * (light_color[:, None, :] * cosine[:, :, None])
    return light #[nb, :, 3]