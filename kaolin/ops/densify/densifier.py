import torch
from .gs_spc_solid import solidify

__all__ = [
    'VolumeDensifier'
]

class VolumeDensifier:
    """ """

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
    def sample_points_in_volume(self, xyz, scale, rotation, opacity, mask=None, count=None):
        """ Samples 3D points inside the approximated volume of a radiance field, represented by Gaussian Splats.
        The gaussians are assumed to reside on the shell of the object. The algorithm will attempt to voxelize
        the gaussians and predict an approximated surface, and then sample additional points within it.

        Args:
            xyz (torch.FloatTensor):
                A tensor of shape (N,3) containing the Gaussian means.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_xyz
            scale (torch.FloatTensor):
                A tensor of shape (N,3) containing the Gaussian covariance scale components, in a format
                of a 3D scale vector per Gaussian. The scale is assumed to be post-activation.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_scaling
            rotation (torch.FloatTensor):
                A tensor of shape (N,4) containing the Gaussian covariance rotation components, in a format
                of a 4D quaternion per Gaussian. The rotation is assumed to be post-activation.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_rotation
            opacity (torch.FloatTensor):
                A tensor of shape (N,1) containing the Gaussian opacities.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_opacity
            mask (optional, torch.BoolTensor):
                An optional (N,) binary mask which selects only a subset of the gaussian to use
                for predicting the shell. Useful if some Gaussians are suspected as noise.
                By default, the mask is assumed to be a tensor of ones to select all Gaussians.
            count (optional, int):
                An optional upper cap on the number of points sampled in the predicted volume.
                If not specified, the volume is evenly sampled in space according to the VolumeDensifier resolution.

        Return:
            (torch.FloatTensor): a tensor of (K, 3) points sampled inside the approximated shape volume.
        """

        # Select the required points and solidify
        num_surface_pts = xyz.shape[0]
        mask = mask.cuda().clone() if mask is not None \
            else xyz.new_ones((num_surface_pts,), device='cuda', dtype=torch.bool)
        # Preprocess: prune low opacity points
        if self.opacity_threshold < 1.0:
            opacity_mask = opacity.reshape(-1) < self.opacity_threshold
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
