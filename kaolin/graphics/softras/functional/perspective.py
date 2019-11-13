import math
import torch


def perspective(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    device = vertices.device
    angle = torch.tensor(angle / 180 * math.pi, dtype=torch.float32, device=device)
    angle = angle[None]
    width = torch.tan(angle)
    width = width[:, None] 
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x,y,z), dim=2)
    return vertices
