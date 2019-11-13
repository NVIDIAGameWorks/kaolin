import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ambient_lighting(light, light_intensity=0.5, light_color=(1,1,1)):
    device = light.device

    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = torch.tensor(light_color, dtype=torch.float32, device=device)
    elif isinstance(light_color, np.ndarray):
        light_color = torch.from_numpy(light_color).float().to(device)
    if light_color.ndimension() == 1:
        light_color = light_color[None, :]

    light += light_intensity * light_color[:, None, :]
    return light #[nb, :, 3]
