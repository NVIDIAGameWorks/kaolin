# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch

from kaolin import _C

__all__ = [
    'sg_diffuse_inner_product',
    'sg_diffuse_fitted',
    'sg_warp_specular_term',
    'fresnel',
    'sg_distribution_term',
    'sg_warp_distribution',
    'cosine_lobe_sg',
    'approximate_sg_integral',
    'sg_irradiance_fitted',
    'sg_irradiance_inner_product',
]

@torch.jit.script
def _dot(a, b):
    """Compute dot product of two tensors on the last axis."""
    return torch.sum(a * b, dim=-1, keepdim=True)

@torch.jit.script
def _reflect(direction, normal):
    """Compute reflection of the vector ``direction`` w.r.t to a normal vector"""
    return direction - 2 * _dot(direction, normal) * normal

@torch.jit.script
def _ggx_v1(m2, nDotX):
    """Helper for computing the Smith visibility term with Trowbridge-Reitz (GGX) distribution"""
    return 1. / (nDotX + torch.sqrt(m2 + (1. - m2) * nDotX * nDotX))

@torch.jit.script
def sg_distribution_term(direction, roughness):
    r"""Returns spherical gaussians approximation of the
    `Trowbridge-Reitz`_ (GGX) distribution used in the Cook-Torrance specular BRDF.

    Use a single lobe to approximate the distribution.

    Args:
        direction (torch.Tensor):
            The normal directions, of shape :math:`(\text{num_points}, 3)`
        roughness (torch.Tensor):
            The roughness of the surface, of shape :math:`(\text{num_points})`
        
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):

            - The amplitude of the spherical gaussians, of shape :math:`(\text{num_points}, 3)`.
            - The input ``direction``.
            - The sharpness of the spherical gaussians, of shape :math:`(\text{num_points})`.

    .. _Trowbridge-Reitz:
        https://opg.optica.org/josa/abstract.cfm?uri=josa-65-5-531
    """
    assert direction.ndim == 2 and direction.shape[-1]
    assert roughness.shape == direction.shape[:1]
    m2 = roughness * roughness
    sharpness = 2. / m2
    amplitude = (1. / (math.pi * m2)).unsqueeze(-1).expand(-1, 3)
    return amplitude, direction, sharpness

@torch.jit.script
def sg_warp_distribution(amplitude, direction, sharpness, view):
    r"""Generate spherical gaussians that best represent the normal distribution function but
    with its axis oriented in the direction of the current BRDF slice.

    Uses the warping operator from `Wang et al`_.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians to be warped,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians to be warped,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians to be warped,
            of shape :math:`(\text{num_sg},)`.
        view (torch.Tensor): The view direction, of shape :math:`(\text{num_sg}, 3)`.
        
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):

            - The input ``amplitude``
            - The warped direction, of shape :math:`(\text{num_sg}, 3)`
            - The warped sharpness, of shape :math:`(\text{num_sg})`

    .. _Wang et al:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2009/12/sg.pdf
    """
    assert amplitude.ndim == 2 and amplitude.shape[-1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert view.shape == amplitude.shape
    warp_direction = _reflect(-view, direction)
    # TODO(cfujitsang): DIBR++ don't apply clamping, is that important?
    warp_sharpness = sharpness / (
        4. * torch.clamp(_dot(direction, view).squeeze(-1), min=1e-4))
    return amplitude, warp_direction, warp_sharpness

@torch.jit.script
def fresnel(ldh, spec_albedo):
    powTerm = torch.pow((1. - ldh), 5)
    return spec_albedo + (1. - spec_albedo) * powTerm

def sg_warp_specular_term(amplitude, direction, sharpness, normal,
                          roughness, view, spec_albedo):
    r"""Computes the specular reflectance from a spherical gaussians lobes representing incoming radiance,
    using the Cook-Torrance microfacet specular shading model.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg},)`.
        normal (torch.Tensor):
            The normal of the surface points where the specular reflectance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.
        roughness (torch.Tensor):
            The roughness of the surface points where the specular reflectance is to be estimated,
            of shape :math:`(\text{num_points})`.
        view (torch.Tensor):
            The direction toward the camera from the surface points where
            the specular reflectance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.
        spec_albedo (torch.Tensor):
            The specular albedo (RGB color) of the surface points where the specular reflectance
            is to be estimated, of shape :math:`(\text{num_points}, 3)`.
    
    Returns:
        (torch.Tensor): The specular reflectance, of shape :math:`(\text{num_points}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[-1]
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert normal.ndim == 2 and normal.shape[-1] == 3
    assert roughness.shape == normal.shape[:1]
    assert view.shape == normal.shape
    assert spec_albedo.shape == normal.shape
    ndf_amplitude, ndf_direction, ndf_sharpness = sg_distribution_term(
        normal, roughness)
    ndf_amplitude, ndf_direction, ndf_sharpness = sg_warp_distribution(
        ndf_amplitude, ndf_direction, ndf_sharpness, view
    )
    ndl = torch.clamp(_dot(normal, ndf_direction), min=0., max=1.)
    ndv = torch.clamp(_dot(normal, view), min=0., max=1.)
    h = ndf_direction + view
    h /= torch.sqrt(_dot(h, h))
    ldh = torch.clamp(_dot(ndf_direction, h), min=0., max=1.)
    
    output = unbatched_reduced_sg_inner_product(
        ndf_amplitude, ndf_direction, ndf_sharpness,
        amplitude, direction, sharpness)
    m2 = (roughness * roughness).unsqueeze(-1)
    output *= _ggx_v1(m2, ndl) * _ggx_v1(m2, ndv)
    output *= fresnel(ldh, spec_albedo)
    output *= ndl
    return torch.clamp(output, min=0.)

@torch.jit.script
def cosine_lobe_sg(direction):
    r'''Returns an approximation of the clamped cosine lobe represented as a spherical gaussian.

    This is to be used with normal of surfaces to apply Lambert's cosine law.

    Args:
        direction (torch.tensor): The direction of the desired lobe, of last dimension 3

    Returns:
        (torch.tensor, torch.Tensor, torch.Tensor):

            - The amplitude of the spherical gaussian, of same shape as ``direction``
            - The input ``direction``
            - The sharpness of the spherical gaussian, of shape ``direction.shape[:-1]``
    '''
    amplitude = torch.full_like(direction, 1.17)
    sharpness = torch.full_like(direction[:, 0], 2.133)

    return amplitude, direction, sharpness

@torch.jit.script
def approximate_sg_integral(amplitude, sharpness):
    r"""Computes an approximate integral of a spherical gaussian over the entire sphere.
    
    The error vs the non-approximate version decreases as sharpness increases.

    Args:
        amplitude (torch.Tensor): The amplitude of the spherical gaussian.
        sharpness (torch.Tensor): The sharpness of the spherical gaussian.

    Returns:
        (torch.tensor): The integral of same shape than ``amplitude``.
    """
    return 2. * math.pi * (amplitude / sharpness.unsqueeze(-1))

@torch.jit.script
def sg_irradiance_fitted(amplitude, direction, sharpness, normal):
    r"""Computes an approximate incident irradiance from multiple spherical gaussians
    representing the incoming radiance.

    The result is broadcasted per point per spherical gaussian.

    .. note::
       The irradiance is computed using a fitted approximation polynomial,
       this approximation were provided by Stephen Hill.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg},)`.
        normal (torch.Tensor):
            The normal of the surface points where the irradiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor):
            The irradiance for each spherical gaussian for each surface point,
            of shape :math:`(\text{num_points}, \text{num_sg}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[-1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert normal.ndim == 2 and normal.shape[1] == 3
    mu_n = torch.einsum('ik,jk->ij', normal, direction);
    lbda = sharpness.unsqueeze(0);

    c0 = 0.36;
    c1 = 1. / (4. * c0);

    eml = torch.exp(-lbda);
    em2l = eml * eml;
    rl = 1. / lbda;

    scale = 1. + 2. * em2l - rl;
    bias = (eml - em2l) * rl - em2l;

    x = torch.sqrt(1. - scale);
    x0 = c0 * mu_n;
    x1 = (c1 * x);
    n = x0 + x1;
    y = torch.where(abs(x0) <= x1,
                    n * n / x,
                    torch.clamp(mu_n, min=0., max=1.))

    result = scale * y + bias;
    return result.unsqueeze(-1) * \
        approximate_sg_integral(amplitude, sharpness).unsqueeze(0);

@torch.jit.script
def sg_diffuse_fitted(amplitude, direction, sharpness, normal, albedo):
    r"""Computes the outgoing radiance from multiple spherical gaussians representing incoming radiance,
    using a Lambertian diffuse BRDF.


    .. note::
       The irradiance is computed using a fitted approximation polynomial,
       this approximation were provided by Stephen Hill. See :func:`sg_irradiance_fitted`.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg},)`.
        normal (torch.Tensor):
            The normal of the surface points where the radiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.
        albedo (torch.Tensor):
            The albedo (RGB color) of the surface points where the radiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor): The diffuse radiance, of shape :math:`(\text{num_points}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert normal.ndim == 2 and normal.shape[1] == 3
    assert albedo.shape == normal.shape
    brdf = albedo / math.pi
    return torch.clamp(
        sg_irradiance_fitted(amplitude, direction, sharpness, normal).mean(1),
        min=0.) * brdf;

def sg_irradiance_inner_product(amplitude, direction, sharpness, normal):
    r"""Computes the approximate incident irradiance from multiple spherical gaussians representing incoming radiance.

    The clamped cosine lobe is approximated as a spherical gaussian,
    and convolved with the incoming radiance lobe using a spherical gaussian inner product.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg},)`.
        normal (torch.Tensor):
            The normal of the surface points where the radiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor): The irradiance, of shape :math:`(\text{num_points}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert normal.ndim == 2 and normal.shape[1] == 3
    lobe_amplitude, lobe_direction, lobe_sharpness = cosine_lobe_sg(normal)
    return torch.clamp(unbatched_reduced_sg_inner_product(
        lobe_amplitude, lobe_direction, lobe_sharpness,
        amplitude, direction, sharpness
    ), min=0.)

def sg_diffuse_inner_product(amplitude, direction, sharpness, normal, albedo):
    r"""Computes the outgoing radiance from multiple spherical gaussians representing incoming radiance,
    using a Lambertian diffuse BRDF.

    This is the diffuse reflectance used in
    `DIB-R++\: Learning to Predict Lighting and Material with a Hybrid Differentiable Renderer`_
    NeurIPS 2021.

    Args:
        amplitude (torch.Tensor):
            The amplitudes of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        direction (torch.Tensor):
            The directions of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg}, 3)`.
        sharpness (torch.Tensor):
            The sharpness of the spherical gaussians representing the incoming radiance,
            of shape :math:`(\text{num_sg},)`.
        normal (torch.Tensor):
            The normal of the surface points where the radiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.
        albedo (torch.Tensor):
            The albedo of the surface points where the radiance is to be estimated,
            of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor): The diffuse radiance, of shape :math:`(\text{num_points}, 3)`.

    .. _DIB-R++\: Learning to Predict Lighting and Material with a Hybrid Differentiable Renderer:
        https://nv-tlabs.github.io/DIBRPlus/
    """
    assert amplitude.ndim == 2 and amplitude.shape[1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert normal.ndim == 2 and normal.shape[1] == 3
    assert albedo.shape == normal.shape
    brdf = albedo / math.pi
    return sg_irradiance_inner_product(
        amplitude, direction, sharpness, normal) * brdf;

@torch.jit.script
def unbatched_sg_inner_product(amplitude, direction, sharpness,
                               other_amplitude, other_direction, other_sharpness):
    r"""Spherical Gaussians Inner Product

    Args:
        amplitude (torch.FloatTensor): amplitude of left hand-side sg,
                                       of shape :math:(\text{num_sg}, 3)`.
        direction (torch.FloatTensor): direction of left hand-side sg,
                                       of shape :math:(\text{num_sg}, 3)`.
        sharpness (torch.FloatTensor): sharpness of left hand-side sg,
                                       of shape :math:(\text{num_size})`.
        other_amplitude (torch.FloatTensor): amplitude of right hand-side sg,
                                             of shape :math:`(\text{num_other}, 3)`.
        other_direction (torch.FloatTensor): direction of right hand-side sg,
                                             of shape :math:`(\text{num_other}, 3)`.
        other_sharpness (torch.FloatTensor): sharpness of right hand-size sg,
                                             of shape :math:`(\text{num_other})`.

    Return:
        (torch.FloatTensor): The inner product,
                             of shape :math:`(\text{num_sg}, \text{num_other}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert other_amplitude.ndim == 2 and other_amplitude.shape[1] == 3
    assert other_direction.shape == other_amplitude.shape
    assert other_sharpness.shape == other_amplitude.shape[:1]
    # TODO(cfujitsang): Maybe reuse reduced cuda code for that.
    num_sg = amplitude.shape[0]
    num_other = other_amplitude.shape[0]
    amplitude = amplitude.reshape(num_sg, 1, 3)
    direction = direction.reshape(num_sg, 1, 3)
    sharpness = sharpness.reshape(num_sg, 1, 1)
    other_amplitude = other_amplitude.reshape(1, num_other, 3)
    other_direction = other_direction.reshape(1, num_other, 3)
    other_sharpness = other_sharpness.reshape(1, num_other, 1)
    # broadcast operations
    # um.shape: (num_other, num_sg, 3)
    #dm = (sharpness * direction + other_sharpness * other_direction).norm(dim=2, keepdim=True)
    dm = (sharpness * direction + other_sharpness * other_direction)
    dm = torch.sqrt(_dot(dm, dm))
    # lm.shape: (num_other, num_sg, 1)
    lm = sharpness + other_sharpness
    # exp.shape: (num_other, num_sg, 3)
    expo = torch.exp(dm - lm) * (amplitude * other_amplitude)
    # other.shape: (num_other, num_sg, 1)
    other = 1.0 - torch.exp(-2.0 * dm)
    # output.shape: (num_other, num_sg, 3)
    return 2.0 * math.pi * expo * other / dm

class UnbatchedReducedSgInnerProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, amplitude, direction, sharpness,
                other_amplitude, other_direction, other_sharpness):
        amplitude = amplitude.contiguous()
        direction = direction.contiguous()
        sharpness = sharpness.contiguous()
        other_amplitude = other_amplitude.contiguous()
        other_direction = other_direction.contiguous()
        other_sharpness = other_sharpness.contiguous()
        ctx.save_for_backward(amplitude, direction, sharpness,
                              other_amplitude, other_direction, other_sharpness)
        output = _C.render.sg.unbatched_reduced_sg_inner_product_forward_cuda(
                amplitude, direction, sharpness,
                other_amplitude, other_direction, other_sharpness)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        amplitude, direction, sharpness, \
        other_amplitude, other_direction, other_sharpness = \
            ctx.saved_tensors
        grad_out = grad_out.contiguous()
        output = _C.render.sg.unbatched_reduced_sg_inner_product_backward_cuda(
            grad_out, amplitude, direction, sharpness,
            other_amplitude, other_direction, other_sharpness)

        return tuple(output)

def unbatched_reduced_sg_inner_product(amplitude, direction, sharpness,
                                       other_amplitude, other_direction, other_sharpness):
    r"""Fused unbatched_sg_inner_product(...).sum(1).

    By being fused it is faster and consume less memory, especially at scale.

    Args:
        amplitude (torch.FloatTensor): amplitude of left hand-side sg,
                                       of shape :math:(\text{num_sg}, 3)`.
        direction (torch.FloatTensor): direction of left hand-side sg,
                                       of shape :math:(\text{num_sg}, 3)`.
        sharpness (torch.FloatTensor): sharpness of left hand-side sg,
                                       of shape :math:(\text{num_size})`.
        other_amplitude (torch.FloatTensor): amplitude of right hand-side sg,
                                             of shape :math:`(\text{num_other}, 3)`.
        other_direction (torch.FloatTensor): direction of right hand-side sg,
                                             of shape :math:`(\text{num_other}, 3)`.
        other_sharpness (torch.FloatTensor): sharpness of right hand-size sg,
                                             of shape :math:`(\text{num_other})`.

    Return:
        (torch.FloatTensor): a reduced output, of shape :math:`(\text{num_sg}, 3)`.
    """
    assert amplitude.ndim == 2 and amplitude.shape[1] == 3
    assert direction.shape == amplitude.shape
    assert sharpness.shape == amplitude.shape[:1]
    assert other_amplitude.ndim == 2 and other_amplitude.shape[1] == 3
    assert other_direction.shape == other_amplitude.shape
    assert other_sharpness.shape == other_amplitude.shape[:1]
    if other_amplitude.shape[0] >= 8:
        output = UnbatchedReducedSgInnerProduct.apply(
            amplitude, direction, sharpness,
            other_amplitude, other_direction, other_sharpness
        )
    else:
        output = unbatched_sg_inner_product(
            amplitude, direction, sharpness,
            other_amplitude, other_direction, other_sharpness
        ).sum(1)
    return output
