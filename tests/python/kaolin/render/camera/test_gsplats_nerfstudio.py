# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
import copy
import math
import os
import random
import pytest
import torch.cuda

import kaolin.io
from kaolin.render.camera import kaolin_camera_to_gsplat_nerfstudio, gsplat_nerfstudio_camera_to_kaolin
from kaolin.render.camera import CameraExtrinsics, PinholeIntrinsics, Camera
from kaolin.utils.bundled_data import SCANNED_TOYS_PATH, SCANNED_TOYS_NAMES, download_scanned_toys_dataset
from kaolin.utils.env_vars import KaolinTestEnvVars
from kaolin.utils.testing import contained_torch_equal, with_seed

import gsplat

TEST_SCANNED_TOYS = os.getenv(KaolinTestEnvVars.TEST_SCANNED_TOYS)

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples', 'render', 'camera', 'conversions', 'gsplat')

def samples_data_path(*args):
    return os.path.join(__samples_path, *args)


def _random_kaolin_camera(dims):
    eye = torch.rand(3) * 2 - 1
    at = torch.rand(3) * 0.5
    up = torch.tensor([0., 0., 1.])
    fov = 0.3 + torch.rand(1).item() * 2.2
    cam = Camera.from_args(eye=eye, at=at, up=up, fov=fov, **dims)
    return cam


@pytest.mark.parametrize('device', [
        'cpu',
        pytest.param('cuda', marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='CUDA not available')),
    ])
class TestCameraConversionRoundTrip:
    @with_seed(42)
    @pytest.mark.parametrize('batched', [True, False])
    def test_roundtrip(self, device, batched):

        def random_dims():
            return {'height': random.randint(200, 800), 'width': random.randint(200, 800),
                    'near': 1e-2 * random.random(), 'far': 1e10 * random.random()}

        shared_dims = random_dims()

        camera_list = []
        for _ in range(3):
            dims = shared_dims if batched else random_dims()
            cam = _random_kaolin_camera(dims).to(device)
            camera_list.append(cam)

        for _ in range(3):
            dims = shared_dims if batched else random_dims()
            eye = torch.rand(3) * 2 - 1
            at = torch.rand(3) * 0.5
            up = torch.tensor([0., 0., 1.])
            focal_x = 200 + torch.rand(1).item() * 800
            focal_y = 200 + torch.rand(1).item() * 800
            cam = Camera(
                extrinsics=CameraExtrinsics.from_lookat(eye=eye, at=at, up=up),
                intrinsics=PinholeIntrinsics.from_focal(focal_x=focal_x, focal_y=focal_y, **dims)
            ).to(device)
            camera_list.append(cam)


        if batched:
            kaolin_input = Camera.cat(camera_list)
            kaolin_input_as_dict = [c.as_dict() for c in kaolin_input]
            gsplat_converted = kaolin_camera_to_gsplat_nerfstudio(kaolin_input)
            kaolin_converted = gsplat_nerfstudio_camera_to_kaolin(**gsplat_converted)
            gsplat_converted2 = kaolin_camera_to_gsplat_nerfstudio(kaolin_converted)
            kaolin_converted_as_dict = [c.as_dict() for c in kaolin_converted]
        else:
            kaolin_input_as_dict = [c.as_dict() for c in camera_list]
            gsplat_converted = [kaolin_camera_to_gsplat_nerfstudio(c) for c in camera_list]
            kaolin_converted = [gsplat_nerfstudio_camera_to_kaolin(**c) for c in gsplat_converted]
            gsplat_converted2 = [kaolin_camera_to_gsplat_nerfstudio(c) for c in kaolin_converted]
            kaolin_converted_as_dict = [c.as_dict() for c in kaolin_converted]

        assert contained_torch_equal(gsplat_converted, gsplat_converted2, approximate=True)
        assert contained_torch_equal(kaolin_input_as_dict, kaolin_converted_as_dict, approximate=True)

    @with_seed(42)
    def test_guess_width_height(self, device):
        dims = {'height': random.randint(200, 800) * 2, 'width': random.randint(200, 800) * 2}
        kaolin_input = _random_kaolin_camera(dims).to(device)
        kaolin_input_as_dict = [c.as_dict() for c in kaolin_input]

        gsplat_converted = kaolin_camera_to_gsplat_nerfstudio(kaolin_input)
        kaolin_converted = gsplat_nerfstudio_camera_to_kaolin(**gsplat_converted)
        kaolin_converted_as_dict = [c.as_dict() for c in kaolin_converted]

        # Test guessing width/height from other properties
        del gsplat_converted['width']
        del gsplat_converted['height']
        kaolin_converted2 = gsplat_nerfstudio_camera_to_kaolin(**gsplat_converted)
        kaolin_converted2_as_dict = [c.as_dict() for c in kaolin_converted2]

        assert contained_torch_equal(kaolin_input_as_dict, kaolin_converted_as_dict, approximate=True)
        assert contained_torch_equal(kaolin_input_as_dict, kaolin_converted2_as_dict, approximate=True)


def _render_with_gsplats(gsmodel, cam_params):
    render_colors, render_alphas, info = gsplat.rendering.rasterization(
        gsmodel['positions'],  # [N, 3]
        gsmodel['orientations'],  # [N, 4]
        gsmodel['scales'],  # [N, 3]
        gsmodel['opacities'],  # [N]
        gsmodel['sh_coeff'],  # [N, S, 3]
        sh_degree=3,  # TODO: fix HACK with gsmodel.sh_degree once we have gaussians class
        **cam_params)
    return render_colors, render_alphas


@pytest.mark.skipif(
    TEST_SCANNED_TOYS is None,
    reason="'KAOLIN_TEST_SCANNED_TOYS' environment variable is not set (will download files if needed).",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available for rendering.")
class TestCameraConversionRenderParity:
    @pytest.fixture(scope='class', autouse=True)
    def download_toys_dataset(self):
        download_scanned_toys_dataset()

    @pytest.mark.parametrize('batched', [True, False])
    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_render_parity(self, toy_name, batched):
        device = 'cuda'
        # create two Kaolin cameras
        cam1_dims = {'height': 480, 'width': 640}
        cam2_dims = cam1_dims if batched else {'height': 300, 'width': 600}  # Can't have varied dims in a batch

        camera_list = [kaolin.render.camera.Camera.from_args(
            eye=torch.tensor([-1, 0.5, 0.5]), at=torch.tensor([0.0, 0.0, 0.3]), up=torch.tensor([0., 0., 1.]),
            fov=math.pi * 45 / 180, **cam1_dims).to(device),
                       kaolin.render.camera.Camera(
            extrinsics=CameraExtrinsics.from_lookat(
                eye=torch.tensor([0.6, 0.6, 0.6]), at=torch.tensor([0.0, 0.0, 0.3]), up=torch.tensor([0., 0., 1.])),
            intrinsics=PinholeIntrinsics.from_focal(focal_x=400, focal_y=800, **cam2_dims)).to(device)]

        # load 3D Gaussian Splat model and aligned low-poly mesh
        mesh_path = os.path.join(SCANNED_TOYS_PATH, f'mesh.{toy_name}.usd')
        splat_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        mesh = kaolin.io.import_mesh(mesh_path).to(device)
        splats = kaolin.io.import_gaussiancloud(splat_path)
        splats = {k: v.to(device) for k, v in splats.items()}  # TODO: remove when we have class

        # render mesh with Kaolin camera
        mesh_silhouettes = []
        for cam in camera_list:
            mesh_rendering = kaolin.render.easy_render.render_mesh(cam, mesh)
            mesh_silhouette = (mesh_rendering[kaolin.render.easy_render.RenderPass.face_idx] >= 0)
            mesh_silhouettes.append(mesh_silhouette)

        splat_silhouettes = []
        opacity_thresh = 0.97
        if batched:
            all_kaolin_cameras = Camera.cat(camera_list)
            all_gsplat_cam_params = kaolin_camera_to_gsplat_nerfstudio(all_kaolin_cameras)
            all_gsplat_renderings = _render_with_gsplats(splats, all_gsplat_cam_params)
            splat_silhouettes = [all_gsplat_renderings[1][c:c+1, ...].squeeze(-1) > opacity_thresh for c in range(len(camera_list))]
        else:
            for cam in camera_list:
                gsplat_cam_params = kaolin_camera_to_gsplat_nerfstudio(cam)
                gsplat_rendering = _render_with_gsplats(splats, gsplat_cam_params)
                splat_silhouette = (gsplat_rendering[1].squeeze(-1) > opacity_thresh)
                splat_silhouettes.append(splat_silhouette)

        def _get_bool_img_precision_recall(img_gt, img_pred):
            recall = img_pred[img_gt].sum() / img_gt.sum()
            precision = img_gt[img_pred].sum() / img_pred.sum()
            return precision, recall

        for i in range(2):
            cam = camera_list[i]
            mesh_silhouette = mesh_silhouettes[i]
            assert 0.03 < mesh_silhouette.float().mean() < 0.8, f'Something is wrong with the chosen camera {i} or mesh rendering'
            assert mesh_silhouette.shape[1] == cam.height
            assert mesh_silhouette.shape[2] == cam.width

            splat_silhouette = splat_silhouettes[i]
            assert 0.03 < splat_silhouette.float().mean() < 0.8, f'Something is wrong with chosen camera {i} or splat rendering'
            assert splat_silhouette.shape[1] == cam.height
            assert splat_silhouette.shape[2] == cam.width

            precision, recall = _get_bool_img_precision_recall(mesh_silhouette, splat_silhouette)
            assert recall > 0.85, f'Failed for {toy_name}'
            assert precision > 0.9, f'Failed for {toy_name}'

        # Also compare with saved ground truth
        gt_cam = torch.load(samples_data_path('camera.pt'), weights_only=True)
        assert contained_torch_equal(gt_cam, camera_list[0].cpu().as_dict(), approximate=True, print_error_context=''), \
            f'Ground truth camera changed, -- please restore test camera to expected parameters'

        mesh_render_path = samples_data_path(f'{toy_name}.mesh_silhouette.png')
        gt_mesh_silhouette = kaolin.io.utils.read_image(mesh_render_path)
        gt_mesh_silhouette = gt_mesh_silhouette.permute(2, 0, 1).to(device) > 0.01

        # Sanity check the mesh silhouette
        precision, recall = _get_bool_img_precision_recall(gt_mesh_silhouette, mesh_silhouettes[0])
        assert recall > 0.97, f'Mesh rendering for {toy_name} does not match ground truth: {mesh_render_path}'
        assert precision > 0.97, f'Mesh rendering for {toy_name} does not match ground truth: {mesh_render_path}'
        del gt_mesh_silhouette

        gt_splat_silhouette_raw = kaolin.io.utils.read_image(samples_data_path(f'{toy_name}.splat_silhouette.png'))
        gt_splat_silhouette = gt_splat_silhouette_raw.permute(2, 0, 1).to(device) > opacity_thresh
        precision, recall = _get_bool_img_precision_recall(gt_splat_silhouette, splat_silhouettes[0])
        assert recall > 0.97, f'Splat rendering for {toy_name} does not match ground truth: {mesh_render_path}'
        assert precision > 0.97, f'Splat rendering for {toy_name} does not match ground truth: {mesh_render_path}'

