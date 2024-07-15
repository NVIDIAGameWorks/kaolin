import torch
from .densify.gs_spc_solid import solidify

class VolumeDensifier:

    def __init__(self, resolution: int=8, opacity_threshold=0.35, post_scale_factor=0.93, jitter=True):
        self.resolution = resolution
        """ A Structured Point Cloud of cubic resolution 2**res will be constructed to fill the volume.
        Higher values require more memory. Max resolution supported is 10.
        """
        self.opacity_threshold = opacity_threshold
        """ Preprocess: if opacity_threshold < 1.0, points with opacity below the threshold will be masked away. """
        self.post_scale_factor = post_scale_factor
        """ Postprocess: if post_scale_factor < 1.0, the returned pointcloud will be rescaled to ensure it fits inside
        the hull of the input points.
        """
        self.jitter = jitter
        """
        If true, applies a small jitter to the returned volume points.
        If false, the returned points lie on an equally distanced grid
        """

    def _jitter(self, pts):
        N = pts.shape[0]
        device = pts.device
        cell_radius = 1.0 / 2.0 **self.resolution
        radius = cell_radius * torch.sqrt(torch.rand(N, device=device))

        # azimuth [0~2pi]
        theta = torch.rand(N, device=device) * 2.0 * torch.pi
        # polar [0~pi]
        phi = torch.rand(N, device=device) * torch.pi

        # spherical coordinates to cartesian
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        delta = torch.stack([x,y,z], dim=1)
        return pts + delta

    @torch.no_grad()
    def sample_points_in_volume(self, model, mask=None, count=None):
        # Select the required points and solidify
        xyz = model.get_xyz.cuda().clone()
        scale = model.get_scaling.cuda().clone()
        rotation = model.get_rotation.cuda().clone()
        opacity = model.get_opacity.cuda().clone()
        num_surface_pts = xyz.shape[0]
        mask = mask.cuda().clone() if mask is not None else xyz.new_ones((num_surface_pts,), device='cuda')
        # Preprocess: prune low opacity points
        if self.opacity_threshold < 1.0:
            opacity_mask = model.get_opacity[:, 0] < self.opacity_threshold
            mask &= ~opacity_mask
        xyz = xyz[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        opacity = opacity[mask]
        volume_pts = solidify(xyz.contiguous(), scale.contiguous(), rotation.contiguous(), opacity.contiguous(),
                              gs_level=self.resolution, query_level=self.resolution)
        if self.jitter:
            volume_pts = self._jitter(volume_pts)
        # Postprocess: rescale returned points to ensure they fit in input points hull
        if self.post_scale_factor < 1.0:
            mean = volume_pts.mean(dim=0)
            volume_pts -= mean
            volume_pts *= self.post_scale_factor
            volume_pts += mean

        num_total_pts = volume_pts.shape[0]
        if count is not None and count < num_total_pts:
            sample_indices = torch.randperm(num_total_pts)[:count]
            volume_pts = volume_pts[sample_indices]
        return volume_pts
