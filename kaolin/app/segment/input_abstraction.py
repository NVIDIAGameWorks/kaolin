from __future__ import annotations

import copy
from abc import ABC
from enum import Enum

import torch

from kaolin.rep import GaussianSplatModel
from kaolin.render.camera import Camera



# @dataclass
# class AggregationSettings:
#     resolution_cap
#     num_views
#     how_to_select_tracking_vies
#     any_othr_Stuff
#
# def auto_aggregate_masks(
#         cloud: CloudAbstraction,
#         cameras,
#         ref_images,  # or just call cloud.render()
#         ref_masks,
#         aggregation_settings: AggregationSettings):
#     tracking_segmenter = cloud.create_tracking_helper()  # Your custom tracking class
#     if tracking_segmenter is None:
#         raise 'Cannot Run Tracking'


DEFAULT_SELECTION_COLOR = (247 / 255., 255 / 255., 88 / 255., 1.0)


class CloudAbstraction(ABC):
    """
    Input abstractions that encompasses soups of renderable items. We don't easily extend to modifying / splitting / etc.
    entities.
    """

    class RenderPass(str, Enum):
        render = "render"
        depth = "depth"
        first_hits = "first_hits"

    def __len__(self):
        pass

    @property
    def device(self):
        ...

    def render(self, camera: Camera, render_passes=None):
        ...

    def with_visibility(self, mask) -> CloudAbstraction:
        """ Only sets elements that are true in the mask to visible. """
        ...
        # for gaussians: toggle_off_gspalts(self.gm, ~mask)
        # self.gm.opacity = copy.deepcopy(self._orig_opacity.detach())

    def project_means_to_2d(self, camera):
        ...

    def with_selection_highlighted(self, mask, color=DEFAULT_SELECTION_COLOR) -> CloudAbstraction:
        ...

    # def create_tracking_aggregation_helper(self) -> TrackingAggregationHelper:
    #     ...


import gsplat
import kaolin.render.camera.gsplats_nerfstudio

class GaussianSplatInput(CloudAbstraction):
    def __init__(self, gsmodel: GaussianSplatModel):
        self._gsmodel = gsmodel
        self._orig_opacities = copy.deepcopy(gsmodel.opacities)
        self._orig_sh_coeff = copy.deepcopy(gsmodel.sh_coeff)

    def __len__(self):
        return len(self.gsmodel)

    @property
    def gsmodel(self):
        return self._gsmodel

    @property
    def device(self):
        return self.gsmodel.positions.device

    def render(self, camera: Camera, render_passes=None):
        # This is RGB
        # TODO: use passes, implement other passes
        gsplat_cam_params = kaolin.render.camera.gsplats_nerfstudio.kaolin_camera_to_gsplat_nerfstudio(camera)
        render_colors, render_alphas, info = gsplat.rendering.rasterization(
            self.gsmodel.positions,
            self.gsmodel.orientations,
            self.gsmodel.scales,
            self.gsmodel.opacities,
            self.gsmodel.sh_coeff,
            sh_degree=self.gsmodel.sh_degree,
            **gsplat_cam_params)

        result = {CloudAbstraction.RenderPass.render: render_colors}
        return result

    def with_visibility(self, mask) -> GaussianSplatInput:
        result = copy.copy(self)  # Not deep
        result._gsmodel.opacities = copy.deepcopy(self._orig_opacities)
        num_enabled = mask.sum()
        if num_enabled == 0:
            result._gsmodel.opacities[...] = 0.0  # post-activation values are stored
        else:
            result._gsmodel.opacities[~mask] = 0.0
        return result

    def project_means_to_2d(self, camera: Camera):
        return camera.transform(self.gsmodel.positions)  # TODO: check does the same thing as before

    def with_selection_highlighted(self, mask, color=DEFAULT_SELECTION_COLOR, weight=0.7):
        """Tint the DC color of the selected gaussians toward ``color``.

        Args:
            mask: boolean tensor over all gaussians; True entries are highlighted.
            color: RGBA highlight color in 0..1 (alpha is ignored for the DC tint).
            weight: blend weight of the highlight color vs. the original color.
        """
        result = copy.copy(self)  # Not deep
        result._gsmodel = copy.copy(self._gsmodel)  # isolate so sh_coeff/opacity writes don't mutate self
        result._gsmodel.sh_coeff = copy.deepcopy(self._orig_sh_coeff)  # Original colors

        num_highlighted = mask.sum()
        if num_highlighted == 0:
            return result

        # We will restore visibility in the highlighted part
        result._gsmodel.opacities = copy.deepcopy(self._gsmodel.opacities)  # Existing opacities
        result._gsmodel.opacities[mask] = self._orig_opacities[mask]

        # SH DC <-> RGB relationship: rgb = 0.5 + C0 * dc.
        C0 = 0.28209479177387814
        new_sh_coeff = copy.deepcopy(self._orig_sh_coeff)
        orig_dc = self._orig_sh_coeff[:, 0, :]
        orig_rgb = orig_dc * C0 + 0.5

        mask = mask.to(orig_rgb.device)
        sel_rgb = torch.tensor(color[:3], dtype=orig_rgb.dtype, device=orig_rgb.device).reshape(1, 3)
        new_rgb = orig_rgb.clone()
        new_rgb[mask] = orig_rgb[mask] * (1.0 - weight) + sel_rgb * weight

        new_sh_coeff[:, 0, :] = (new_rgb - 0.5) / C0
        result._gsmodel.sh_coeff = new_sh_coeff
        return result

    # INRIA SPLATS REFERENCE
    # def highlight_gsplat_selection(features_dc: torch.Tensor, orig_colors: torch.Tensor, mask: torch.BoolTensor,
    #                                selection_color: torch.Tensor, weight=0.7):
    #     """
    #     Highlights selected gaussians with a color.
    #
    #     Args:
    #         orig_colors: of all gaussians, e.g. compute once with get_colors above
    #         mask:
    #         selection_color:
    #         weight: weight to assign to selection color vs. orig color
    #
    #     Returns:
    #
    #     """
    #     new_colors = orig_colors.clone()
    #     new_colors[mask, :] = orig_colors[mask, :] * (1 - weight) + selection_color[..., :3].reshape(1, 3) * weight
    #
    #     # Now modify the features accordingly
    #     fused_color = RGB2SH(new_colors)
    #     features_dc[:, 0, :] = fused_color