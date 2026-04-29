import logging

import torch
import torch.nn.functional as F
import gsplat.rendering

import kaolin
from kaolin.utils.log import log_tensor
from kaolin.visualize.ipython import quick_viz
from kaolin.app.segment.selection import CloudSelection  # TODO: obviously should live elsewhere

logger = logging.getLogger(__name__)



def gaussians_in_mask(kaolin_cam, gaussians_pos, mask):
    vertices_camera = kaolin_cam.transform(gaussians_pos)
    new_selection = CloudSelection.from_image_mask(vertices_camera, mask)
    return new_selection.mask


def train_splat_subset_colors(gsmodel, mask,
                              kal_cam, image, image_mask,
                              log_every_n=10,
                              lr=1.5e-2,
                              n_steps=200):
    """Optimize the DC (albedo) channel of `sh_coeff` for the masked splats so the
    rendered image matches the (image, image_mask) target. Higher-order SH channels
    are kept frozen. Returns the updated DC channel as a `(N, 1, 3)` tensor.
    """
    cam_params = kaolin.render.camera.kaolin_camera_to_gsplat_nerfstudio(kal_cam)
    image_mask = image_mask.to(gsmodel.positions.device)

    # Split sh_coeff into DC (albedo) and higher-order (specular) channels.
    # sh_coeff is (N, S, 3) where S = (sh_degree + 1) ** 2; channel 0 is the DC term.
    base_albedo = gsmodel.sh_coeff[:, 0:1, :].detach()        # (N, 1, 3)
    specular = gsmodel.sh_coeff[:, 1:, :].detach()            # (N, S-1, 3), frozen
    mask_indices = mask.nonzero(as_tuple=False).squeeze(-1)   # (M,) indices where mask is True
    trainable_albedo = torch.nn.Parameter(base_albedo[mask_indices].clone())  # (M, 1, 3)

    optimizer = torch.optim.Adam(
        [{"params": trainable_albedo, "lr": lr, "name": "trainable_albedo"}],
        eps=1e-10, betas=(0.9, 0.999))

    pixels = None
    for step in range(n_steps):
        # Rebuild sh_coeff with the optimized albedo spliced into the DC channel.
        # `index_copy` is out-of-place so gradients flow back through trainable_albedo.
        cur_albedo = base_albedo.index_copy(0, mask_indices, trainable_albedo)  # (N, 1, 3)
        cur_sh_coeff = torch.cat([cur_albedo, specular], dim=1)                 # (N, S, 3)

        colors, render_alphas, info = gsplat.rendering.rasterization(
            gsmodel.positions,
            gsmodel.orientations,
            gsmodel.scales,
            gsmodel.opacities,
            cur_sh_coeff,
            sh_degree=gsmodel.sh_degree,
            **cam_params
        )
        if step == 0:
            pixels = colors.detach().clone()  # (1, H, W, 3) in 0..1
            pixels[image_mask] = image.to(gsmodel.positions.device)[image_mask]
            log_tensor(pixels, 'ground_truth pixels', logger, print_stats=True)
            log_tensor(colors, 'rendered_colors', logger, print_stats=True)
            quick_viz(colors.permute(0, 3, 1, 2), inches=6)
            quick_viz(pixels.permute(0, 3, 1, 2), inches=6)

        loss = F.l1_loss(colors, pixels)
        loss.backward()

        if step % log_every_n == 0:
            logger.info(f'Loss {step} / {n_steps}: {loss.item()}')

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return base_albedo.index_copy(0, mask_indices, trainable_albedo.detach())