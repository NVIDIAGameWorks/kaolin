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


import torch
import pytest
import os
import kaolin as kal
from kaolin.utils.testing import with_seed, check_allclose

from kaolin.physics.simplicits import SimplicitsObject, SkinnedPhysicsPoints, SkinnedPoints, PhysicsPoints, SkinningModule, SimplicitsMLP

import logging

########### Global Settings ##############

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


########### Test Simplicits Easy API Training ##############
class TestSimplicitsObject:
    
    @pytest.fixture(autouse=True, scope='class')
    def test_dir(self,):
        return os.path.dirname(os.path.realpath(__file__))

    @pytest.fixture(autouse=True, scope='class')
    def samples_path(self, test_dir):
        return os.path.join(test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')

    @pytest.fixture(autouse=True, scope='class')
    def device(self):
        return 'cuda'

    @pytest.fixture(autouse=True, scope='class')
    def dtype(self):
        return torch.float

    def test_create_rigid(self, device, dtype):
        pts = torch.rand(50, 3, device=device, dtype=dtype)
        obj = SimplicitsObject.create_rigid(pts=pts, yms=1e5, prs=0.45, rhos=100.0, appx_vol=1.0)
        assert obj.num_handles == 1
        assert torch.equal(obj.pts, pts)
        # Rigid body has a single constant handle: every point maps to weight 1.0
        weights = obj.skinning_mod.compute_skinning_weights(pts[:5])
        assert weights.shape == (5, 1)
        assert torch.allclose(weights, torch.ones(5, 1, device=device, dtype=dtype))

    def test_ctor_with_base_skinning_weight_function(self, device, dtype):
        pts = torch.rand(50, 3, device=device, dtype=dtype)
        # base function returns 2 weights → total num_handles = 3 (2 + 1 constant)
        def base_fn(x):
            return torch.ones(x.shape[0], 2, device=x.device, dtype=x.dtype) / 2.0
        obj = SimplicitsObject(pts=pts, yms=1e5, prs=0.45, rhos=100.0, appx_vol=1.0,
                               skinning_mod=SkinningModule.from_function(base_fn))
        assert obj.num_handles == 3
        assert torch.equal(obj.pts, pts)
        # base_fn returns [0.5, 0.5]; compute_skinning_weights appends constant 1.0 → [0.5, 0.5, 1.0]
        weights = obj.skinning_mod.compute_skinning_weights(pts[:5])
        expected = torch.tensor([0.5, 0.5, 1.0], device=device, dtype=dtype).expand(5, -1)
        assert torch.allclose(weights, expected)

    def test_ctor_with_custom_bb(self, device, dtype):
        pts = torch.rand(50, 3, device=device, dtype=dtype)
        bb_min = torch.zeros(3, device=device, dtype=dtype)
        bb_max = torch.ones(3, device=device, dtype=dtype) * 2.0
        def base_fn(x):
            return torch.ones(x.shape[0], 2, device=x.device, dtype=x.dtype) / 2.0
        obj = SimplicitsObject(pts=pts, yms=1e5, prs=0.45, rhos=500.0, appx_vol=1.0,
                               skinning_mod=SkinningModule.from_function(base_fn, bb_min=bb_min, bb_max=bb_max))
        assert obj.num_handles == 3
        # bb_min / bb_max stored correctly on the skinning module
        assert torch.equal(obj.skinning_mod.bb_min.cpu(), bb_min.cpu())
        assert torch.equal(obj.skinning_mod.bb_max.cpu(), bb_max.cpu())
        # _offset_scale uses the custom bb:
        # pts ∈ [0, 1]^3, bb = (0, 2) → scaled = pts / 2 ∈ [0, 0.5]
        scaled = obj.skinning_mod._offset_scale(pts)
        assert scaled.shape == pts.shape
        assert scaled.min().item() >= 0.0 - 1e-6
        assert scaled.max().item() <= 0.5 + 1e-6

    def _make_cpu_simplicits_object(self):
        # Use SimplicitsMLP so the skinning_mod has both buffers (bb_min/bb_max) and
        # real nn.Linear parameters — tests that .to() / .cuda() / .cpu() move both.
        pts = torch.rand(50, 3, dtype=torch.float32, device='cpu')
        skinning_mod = SimplicitsMLP(
            spatial_dimensions=3, layer_width=8, num_handles=4, num_layers=2,
        ).to('cpu', dtype=torch.float32)
        return SimplicitsObject(
            pts=pts, yms=1e5, prs=0.45, rhos=500.0, appx_vol=1.0,
            skinning_mod=skinning_mod,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_to_device_moves_skinning_mod(self):
        obj = self._make_cpu_simplicits_object()
        moved = obj.to('cuda')
        for attr in ['pts', 'yms', 'prs', 'rhos', 'appx_vol']:
            assert getattr(moved, attr).device.type == 'cuda'
        for p in moved.skinning_mod.parameters():
            assert p.device.type == 'cuda'
        for b in moved.skinning_mod.buffers():
            assert b.device.type == 'cuda'
        weights = moved.skinning_mod.compute_skinning_weights(moved.pts)
        assert weights.device.type == 'cuda'
        assert weights.shape == (moved.pts.shape[0], moved.num_handles)

    def test_to_dtype_casts_skinning_mod(self):
        obj = self._make_cpu_simplicits_object()
        moved = obj.to(dtype=torch.float64)
        for attr in ['pts', 'yms', 'prs', 'rhos', 'appx_vol']:
            assert getattr(moved, attr).dtype == torch.float64
        for p in moved.skinning_mod.parameters():
            assert p.dtype == torch.float64
        for b in moved.skinning_mod.buffers():
            assert b.dtype == torch.float64

    def test_to_attributes_filter_leaves_skinning_mod(self):
        obj = self._make_cpu_simplicits_object()
        moved = obj.to(dtype=torch.float64, attributes=['pts'])
        assert moved.pts.dtype == torch.float64
        assert moved.yms.dtype == torch.float32
        for p in moved.skinning_mod.parameters():
            assert p.dtype == torch.float32
        for b in moved.skinning_mod.buffers():
            assert b.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_cpu_roundtrip_moves_skinning_mod(self):
        obj = self._make_cpu_simplicits_object()
        on_cuda = obj.cuda()
        for attr in ['pts', 'yms', 'prs', 'rhos', 'appx_vol']:
            assert getattr(on_cuda, attr).device.type == 'cuda'
        for p in on_cuda.skinning_mod.parameters():
            assert p.device.type == 'cuda'
        for b in on_cuda.skinning_mod.buffers():
            assert b.device.type == 'cuda'

        back_on_cpu = on_cuda.cpu()
        for attr in ['pts', 'yms', 'prs', 'rhos', 'appx_vol']:
            assert getattr(back_on_cpu, attr).device.type == 'cpu'
        for p in back_on_cpu.skinning_mod.parameters():
            assert p.device.type == 'cpu'
        for b in back_on_cpu.skinning_mod.buffers():
            assert b.device.type == 'cpu'

    @pytest.fixture
    @with_seed(0, 0, 0)
    def fox_object(self, device, dtype, samples_path):
        """Fixture to set up fox object for testing."""
        # Import and triangulate to enable rasterization; move to GPU
        mesh = kal.io.import_mesh(os.path.join(
            samples_path, "physics/fox.obj"), triangulate=True).to(device)
        mesh.vertices = kal.ops.pointcloud.center_points(
            mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
        orig_vertices = mesh.vertices.clone()  # Also save original undeformed vertices

        # Physics material parameters
        soft_youngs_modulus = 1e4
        poisson_ratio = 0.45
        rho = 500  # kg/m^3
        appx_vol = 0.5  # m^3

        # Points sampled over the object
        num_samples = 10000
        uniform_pts = torch.rand(num_samples, 3, device=device) * (orig_vertices.max(dim=0).values -
                                                                   orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
        pts = uniform_pts
        yms = torch.full((pts.shape[0],), soft_youngs_modulus, dtype=dtype, device=device)
        prs = torch.full((pts.shape[0],), poisson_ratio, dtype=dtype, device=device)
        rhos = torch.full((pts.shape[0],), rho, dtype=dtype, device=device)
        appx_vol = torch.tensor(appx_vol, dtype=dtype, device=device)
        return pts, yms, prs, rhos, appx_vol, orig_vertices

    @with_seed(0, 0, 0)
    def test_training_loss_matches_reference(self, device, dtype, fox_object):
        logging.disable(logging.INFO)
        try:
            logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

            r"Step 1: Load and Setup Object"
            pts, yms, prs, rhos, appx_vol, orig_vertices = fox_object

            r"Step 2: Create Simplicits Object"
            # Set up logging handler to capture training values
            training_vals = []

            # Create custom handler to store logged values
            class LogCaptureHandler(logging.Handler):
                def emit(self, record):
                    msg = record.getMessage()
                    if 'le:' in msg and 'lo:' in msg:
                        # Extract le and lo values from log message
                        le_start = msg.find('le: ') + 4
                        le_end = msg.find(', lo:')
                        lo_start = msg.find('lo: ') + 4
                        le = float(msg[le_start:le_end])
                        lo = float(msg[lo_start:])
                        training_vals.append((le, lo))

            log_handler = LogCaptureHandler()
            logger = logging.getLogger('kaolin.physics')
            logger.addHandler(log_handler)

            # Create object and capture training logs
            sim_obj = kal.physics.simplicits.SimplicitsObject.create_trained(pts,
                                                                             yms,
                                                                             prs,
                                                                             rhos,
                                                                             appx_vol,
                                                                             num_handles=10,
                                                                             num_samples=1000,
                                                                             model_layers=6,
                                                                             training_batch_size=10,
                                                                             training_num_steps=4000,
                                                                             training_lr_start=0.001,
                                                                             training_lr_end=0.001,
                                                                             training_le_coeff=0.1,
                                                                             training_lo_coeff=1000000,
                                                                             training_log_every=1000,
                                                                             normalize_for_training=True)
        
            # NOTE: Take a look at this MLP if this test fails. Might help with debugging.
            # torch.save(sim_obj.base_skinning_weight_function, os.path.dirname(os.path.realpath(__file__)) + "/regression_test_data/box_reference_weights_fcn_10_handles.pth")

            # Clean up logging handler
            logger.removeHandler(log_handler)
            assert isinstance(sim_obj, kal.physics.simplicits.SimplicitsObject)

            r"Step 3: Read Reference Train Vals"
            filename = os.path.dirname(os.path.realpath(__file__)) + \
                "/regression_test_data/box_training_reference_log_4k_steps.txt"
            reference_training_val = torch.load(filename, weights_only=False)

            i = 0
            r"Step 4: Asserts Training Match"
            for tvals in training_vals:
                le = reference_training_val[i][0]
                lo = reference_training_val[i][1]
                assert (abs(le - tvals[0]) < 0.005 * le)
                assert (abs(lo - tvals[1]) < 0.005 * lo)
                i += 1
        finally:
            logging.disable(logging.NOTSET)

    @with_seed(0, 0, 0)
    def test_training_loss_decrease(self, device, dtype, fox_object):
        logging.disable(logging.INFO)
        try:
            logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

            r"Step 1: Load and Setup Object"
            pts, yms, prs, rhos, appx_vol, orig_vertices = fox_object

            # Create object and capture training logs
            sim_obj = kal.physics.simplicits.SimplicitsObject.create_trained(pts,
                                                                             yms,
                                                                             prs,
                                                                             rhos,
                                                                             appx_vol,
                                                                             num_handles=10,
                                                                             num_samples=1000,
                                                                             model_layers=6,
                                                                             training_batch_size=10,
                                                                             training_num_steps=1000,
                                                                             training_lr_start=0.001,
                                                                             training_lr_end=0.001,
                                                                             training_le_coeff=0.1,
                                                                             training_lo_coeff=1000000,
                                                                             training_log_every=1000,
                                                                             normalize_for_training=True)

            r"Step 3: Read Reference Train Vals"
            filename = os.path.dirname(os.path.realpath(__file__)) + \
                "/regression_test_data/box_training_reference_log_4k_steps.txt"
            reference_training_val = torch.load(filename, weights_only=False)

            r"Step 5: Verify losses are low enough post-training a few thousand steps"
            bb_max = torch.max(pts, dim=0).values
            bb_min = torch.min(pts, dim=0).values
            bb_vol = (bb_max[0] - bb_min[0]) * (bb_max[1] -
                                                bb_min[1]) * (bb_max[2] - bb_min[2])

            # Normalize the appx vol of object
            norm_bb_max = torch.max((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_max of the normalized pts
            norm_bb_min = torch.min((pts - bb_min) / (bb_max - bb_min),
                                    dim=0).values  # get the bb_min of the normalized pts

            norm_bb_vol = (norm_bb_max[0] - norm_bb_min[0]) * (norm_bb_max[1] -
                                                            norm_bb_min[1]) * (norm_bb_max[2] - norm_bb_min[2])
            normalized_pts = (pts - bb_min) / (bb_max - bb_min)
            norm_appx_vol = appx_vol * (norm_bb_vol / bb_vol)

            # Set pts, appx_vol, yms, prs, rhos to normalized values
            training_pts = normalized_pts

            le, lo = kal.physics.simplicits.losses.compute_losses(sim_obj.skinning_mod,
                                                                training_pts,
                                                                yms.unsqueeze(-1),
                                                                prs.unsqueeze(-1),
                                                                rhos.unsqueeze(-1),
                                                                1.0,
                                                                le_coeff=0.1,
                                                                lo_coeff=1000000,
                                                                batch_size=10,
                                                                appx_vol=norm_appx_vol,
                                                                num_samples=1000)

            assert reference_training_val[0][0] > le
            assert reference_training_val[0][1] > lo
        finally:
            logging.disable(logging.NOTSET)

    @pytest.fixture
    def example_object(self, device, dtype):
        """Small rigid SimplicitsObject for bake tests (no heavy training required)."""
        pts = torch.rand(50, 3, device=device, dtype=dtype)
        yms = torch.rand(50, device=device, dtype=dtype)
        prs = torch.rand(50, device=device, dtype=dtype)
        rhos = torch.rand(50, device=device, dtype=dtype)
        appx_vol = torch.tensor([1.0], device=device, dtype=dtype)
        return SimplicitsObject.create_rigid(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)

    @pytest.mark.parametrize("with_renderable_pts", [False, True])
    def test_bake_with_num_qps(self, example_object, with_renderable_pts, device, dtype):
        num_qps = 10
        if with_renderable_pts:
            renderable_pts = torch.rand(20, 3, device=device, dtype=dtype)
        else:
            renderable_pts = None
        baked = example_object.bake(
            num_qps=num_qps,
            renderable_pts=renderable_pts)
        assert baked.pts.shape == (num_qps, 3)
        assert baked.yms.shape == (num_qps,)
        assert baked.prs.shape == (num_qps,)
        assert baked.rhos.shape == (num_qps,)
        assert torch.equal(baked.appx_vol, example_object.appx_vol)
        assert baked.skinning_weights.shape == (num_qps, example_object.num_handles)
        assert baked.dwdx.shape == (num_qps, example_object.num_handles, 3)
        # baked pts must come from the original set (bake subsamples by index)
        matches = (baked.pts.unsqueeze(1) == example_object.pts.unsqueeze(0)).all(dim=2)
        assert matches.any(dim=1).all()
        # weights and dwdx must match what the skinning_mod produces for those pts
        assert torch.equal(baked.skinning_weights,
                           example_object.skinning_mod.compute_skinning_weights(baked.pts))
        assert torch.equal(baked.dwdx,
                           example_object.skinning_mod.compute_dwdx(baked.pts))

        if with_renderable_pts:
            assert isinstance(baked.renderable, SkinnedPoints)
            assert torch.equal(baked.renderable.pts, renderable_pts)
            assert torch.equal(baked.renderable.skinning_weights, example_object.skinning_mod.compute_skinning_weights(renderable_pts))
        else:
            assert baked.renderable is None

    @pytest.mark.parametrize("with_renderable_pts", [False, True])
    def test_bake_with_sampling_indices(self, example_object, with_renderable_pts, device, dtype):
        sampling_indices = torch.randint(0, example_object.pts.shape[0], (10,), device=device, dtype=torch.long)
        if with_renderable_pts:
            renderable_pts = torch.rand(20, 3, device=device, dtype=dtype)
        else:
            renderable_pts = None
        baked = example_object.bake(
            sampling_indices=sampling_indices,
            renderable_pts=renderable_pts)
        assert torch.equal(baked.pts, example_object.pts[sampling_indices])
        assert torch.equal(baked.yms, example_object.yms[sampling_indices])
        assert torch.equal(baked.prs, example_object.prs[sampling_indices])
        assert torch.equal(baked.rhos, example_object.rhos[sampling_indices])
        assert torch.equal(baked.appx_vol, example_object.appx_vol)
        assert torch.equal(baked.skinning_weights, example_object.skinning_mod.compute_skinning_weights(example_object.pts[sampling_indices]))
        expected_dwdx = example_object.skinning_mod.compute_dwdx(example_object.pts[sampling_indices])
        assert torch.equal(baked.dwdx, expected_dwdx)

        if with_renderable_pts:
            assert isinstance(baked.renderable, SkinnedPoints)
            assert torch.equal(baked.renderable.pts, renderable_pts)
            assert torch.equal(baked.renderable.skinning_weights, example_object.skinning_mod.compute_skinning_weights(renderable_pts))
        else:
            assert baked.renderable is None

    def test_bake_no_args_raises(self, example_object):
        with pytest.raises(ValueError, match="bake\\(\\) requires either num_qps or sampling_indices"):
            example_object.bake()

########### Test SkinnedPhysicsPoints ##############
class TestSkinnedPhysicsPoints:

    @pytest.fixture
    def device(self):
        return 'cuda'

    @pytest.fixture
    def dtype(self):
        return torch.float

    @pytest.fixture
    def physics_data(self, device, dtype):
        n = 50
        pts = torch.rand(n, 3, device=device, dtype=dtype)
        yms = torch.full((n,), 1e5, device=device, dtype=dtype)
        prs = torch.full((n,), 0.45, device=device, dtype=dtype)
        rhos = torch.full((n,), 100.0, device=device, dtype=dtype)
        appx_vol = torch.tensor([1.0], device=device, dtype=dtype)
        return pts, yms, prs, rhos, appx_vol

    @pytest.fixture
    def sim_object(self, physics_data):
        pts, yms, prs, rhos, appx_vol = physics_data
        return SimplicitsObject.create_rigid(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)

    def test_init_no_renderable_pts(self, sim_object):
        baked = SkinnedPhysicsPoints.from_skinning_mod(
            pts=sim_object.pts, yms=sim_object.yms, prs=sim_object.prs,
            rhos=sim_object.rhos, appx_vol=sim_object.appx_vol,
            skinning_mod=sim_object.skinning_mod)
        assert baked.renderable is None
        assert torch.equal(baked.pts, sim_object.pts)
        assert torch.equal(baked.yms, sim_object.yms)
        assert torch.equal(baked.prs, sim_object.prs)
        assert torch.equal(baked.rhos, sim_object.rhos)
        assert torch.equal(baked.appx_vol, sim_object.appx_vol)
        assert baked.skinning_weights.shape == (sim_object.pts.shape[0], sim_object.num_handles)
        assert baked.dwdx.shape == (sim_object.pts.shape[0], sim_object.num_handles, 3)

    def test_init_with_renderable_pts(self, sim_object, device, dtype):
        render_pts = torch.rand(15, 3, device=device, dtype=dtype)
        baked = SkinnedPhysicsPoints.from_skinning_mod(
            pts=sim_object.pts, yms=sim_object.yms, prs=sim_object.prs,
            rhos=sim_object.rhos, appx_vol=sim_object.appx_vol,
            skinning_mod=sim_object.skinning_mod)
        obj = SkinnedPhysicsPoints(
            pts=baked.pts, yms=baked.yms, prs=baked.prs, rhos=baked.rhos, appx_vol=baked.appx_vol,
            skinning_weights=baked.skinning_weights, dwdx=baked.dwdx,
            renderable=SkinnedPoints.from_skinning_mod(pts=render_pts, skinning_mod=sim_object.skinning_mod))
        check_allclose(obj.renderable.pts, render_pts)
        assert obj.renderable.skinning_weights.shape == (15, sim_object.num_handles)
        expected_render_weights = sim_object.skinning_mod.compute_skinning_weights(render_pts)
        assert torch.equal(obj.renderable.skinning_weights, expected_render_weights)

    def test_from_skinning_mod_no_renderable_pts(self, sim_object):
        baked = SkinnedPhysicsPoints.from_skinning_mod(
            pts=sim_object.pts, yms=sim_object.yms, prs=sim_object.prs,
            rhos=sim_object.rhos, appx_vol=sim_object.appx_vol,
            skinning_mod=sim_object.skinning_mod)
        assert baked.renderable is None
        assert torch.equal(baked.pts, sim_object.pts)
        assert torch.equal(baked.yms, sim_object.yms)
        assert torch.equal(baked.prs, sim_object.prs)
        assert torch.equal(baked.rhos, sim_object.rhos)
        assert torch.equal(baked.appx_vol, sim_object.appx_vol)
        expected_weights = sim_object.skinning_mod.compute_skinning_weights(sim_object.pts)
        assert torch.equal(baked.skinning_weights, expected_weights)
        assert baked.dwdx.shape == (sim_object.pts.shape[0], sim_object.num_handles, 3)

    def test_from_skinning_mod_with_renderable_pts(self, sim_object, device, dtype):
        render_pts = torch.rand(15, 3, device=device, dtype=dtype)
        baked = SkinnedPhysicsPoints.from_skinning_mod(
            pts=sim_object.pts, yms=sim_object.yms, prs=sim_object.prs,
            rhos=sim_object.rhos, appx_vol=sim_object.appx_vol,
            skinning_mod=sim_object.skinning_mod, renderable_pts=render_pts)
        assert torch.equal(baked.pts, sim_object.pts)
        expected_weights = sim_object.skinning_mod.compute_skinning_weights(sim_object.pts)
        assert torch.equal(baked.skinning_weights, expected_weights)
        assert baked.dwdx.shape == (sim_object.pts.shape[0], sim_object.num_handles, 3)
        check_allclose(baked.renderable.pts, render_pts)
        expected_render_weights = sim_object.skinning_mod.compute_skinning_weights(render_pts)
        assert torch.equal(baked.renderable.skinning_weights, expected_render_weights)


########### Test PhysicsPoints ##############
class TestPhysicsPoints:

    @pytest.fixture
    def device(self):
        return 'cuda'

    @pytest.fixture
    def dtype(self):
        return torch.float

    def test_init_tensor_inputs(self, device, dtype):
        N = 20
        pts = torch.rand(N, 3, device=device, dtype=dtype)
        yms = torch.full((N,), 1e5, device=device, dtype=dtype)
        prs = torch.full((N,), 0.45, device=device, dtype=dtype)
        rhos = torch.full((N,), 500.0, device=device, dtype=dtype)
        appx_vol = torch.tensor(1.0, device=device, dtype=dtype)
        obj = PhysicsPoints(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)
        assert torch.equal(obj.pts, pts)
        assert torch.equal(obj.yms, yms)
        assert torch.equal(obj.prs, prs)
        assert torch.equal(obj.rhos, rhos)
        assert torch.equal(obj.appx_vol, appx_vol)

    def test_init_scalar_inputs(self, device, dtype):
        N = 20
        pts = torch.rand(N, 3, device=device, dtype=dtype)
        obj = PhysicsPoints(pts=pts, yms=1e5, prs=0.45, rhos=500.0, appx_vol=1.0)
        assert torch.equal(obj.pts, pts)
        assert torch.equal(obj.yms, torch.full((N,), 1e5, device=device, dtype=dtype))
        assert torch.equal(obj.prs, torch.full((N,), 0.45, device=device, dtype=dtype))
        assert torch.equal(obj.rhos, torch.full((N,), 500.0, device=device, dtype=dtype))
        assert torch.equal(obj.appx_vol, torch.tensor(1.0, device=device, dtype=dtype))


########### Test SkinnedPoints ##############
class TestSkinnedPoints:

    @pytest.fixture
    def device(self):
        return 'cuda'

    @pytest.fixture
    def dtype(self):
        return torch.float

    def test_init(self, device, dtype):
        pts = torch.rand(10, 3, device=device, dtype=dtype)
        weights = torch.rand(10, 4, device=device, dtype=dtype)
        sp = SkinnedPoints(pts=pts, skinning_weights=weights)
        assert torch.equal(sp.pts, pts)
        assert torch.equal(sp.skinning_weights, weights)
        assert sp.num_handles == 4

    def test_from_skinning_mod(self, device, dtype):
        pts = torch.rand(10, 3, device=device, dtype=dtype)
        sim_obj = SimplicitsObject.create_rigid(pts=pts, yms=1e5, prs=0.45,
                                                rhos=500.0, appx_vol=1.0)
        render_pts = torch.rand(5, 3, device=device, dtype=dtype)
        sp = SkinnedPoints.from_skinning_mod(pts=render_pts, skinning_mod=sim_obj.skinning_mod)
        expected_weights = sim_obj.skinning_mod.compute_skinning_weights(render_pts)
        assert torch.equal(sp.pts, render_pts)
        assert torch.equal(sp.skinning_weights, expected_weights)
