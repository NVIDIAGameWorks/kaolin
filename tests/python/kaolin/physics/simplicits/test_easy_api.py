# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np
import warp as wp
import pytest
import os
import kaolin as kal
from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_allclose

from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene


import logging
logging.disable(logging.INFO)

########### Global Settings ##############

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


########### Test Simplicits Easy API Training ##############
class TestEasyAPISimplicitsObjectTraining:
    
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
        approx_volume = 0.5  # m^3

        # Points sampled over the object
        num_samples = 1000000

        uniform_pts = torch.rand(num_samples, 3, device=device) * (orig_vertices.max(dim=0).values -
                                                                   orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
        boolean_signs = kal.ops.mesh.check_sign(mesh.vertices.unsqueeze(
            0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)
        pts = uniform_pts[boolean_signs.squeeze()]
        yms = torch.full((pts.shape[0],), soft_youngs_modulus, device=device)
        prs = torch.full((pts.shape[0],), poisson_ratio, device=device)
        rhos = torch.full((pts.shape[0],), rho, device=device)
        return pts, yms, prs, rhos, approx_volume, orig_vertices

    @pytest.fixture
    @with_seed(0, 0, 0)
    def box_object(self, device, dtype, samples_path):
        """Fixture to set up box object for testing."""
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
        approx_volume = 0.5  # m^3

        # Points sampled over the object
        num_samples = 10000
        uniform_pts = torch.rand(num_samples, 3, device=device) * (orig_vertices.max(dim=0).values -
                                                                   orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
        pts = uniform_pts
        yms = torch.full((pts.shape[0],), soft_youngs_modulus, device=device)
        prs = torch.full((pts.shape[0],), poisson_ratio, device=device)
        rhos = torch.full((pts.shape[0],), rho, device=device)
        return pts, yms, prs, rhos, approx_volume, orig_vertices

    @with_seed(0, 0, 0)
    def test_training_loss_matches_reference(self, device, dtype, box_object):
        logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

        r"Step 1: Load and Setup Object"
        pts, yms, prs, rhos, approx_volume, orig_vertices = box_object

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
                                                                        torch.tensor(
                                                                            [approx_volume], dtype=dtype, device=device),
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
        # torch.save(sim_obj.skinning_weight_function.model, os.path.dirname(os.path.realpath(__file__)) + "/regression_test_data/box_reference_weights_fcn_10_handles.pth")

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

    @with_seed(0, 0, 0)
    def test_training_loss_decrease(self, device, dtype, box_object):
        logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

        r"Step 1: Load and Setup Object"
        pts, yms, prs, rhos, approx_volume, orig_vertices = box_object

        # Create object and capture training logs
        sim_obj = kal.physics.simplicits.SimplicitsObject.create_trained(pts,
                                                                        yms,
                                                                        prs,
                                                                        rhos,
                                                                        torch.tensor(
                                                                            [approx_volume], dtype=dtype, device=device),
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
        norm_appx_vol = approx_volume * (norm_bb_vol / bb_vol)

        # Set pts, appx_vol, yms, prs, rhos to normalized values
        training_pts = normalized_pts

        le, lo = kal.physics.simplicits.losses.compute_losses(sim_obj.skinning_weight_function.model,
                                                            training_pts,
                                                            yms.unsqueeze(-1),
                                                            prs.unsqueeze(-1),
                                                            rhos.unsqueeze(-1),
                                                            1.0,
                                                            le_coeff=0.1,
                                                            lo_coeff=1000000,
                                                            batch_size=10,
                                                            num_handles=10,
                                                            appx_vol=norm_appx_vol,
                                                            num_samples=1000)

        assert reference_training_val[0][0] > le
        assert reference_training_val[0][1] > lo

class TestEasyAPISimplicitsScene:
    
    @pytest.fixture(autouse=True, scope='class')
    def device(self):
        return 'cuda'
    
    @pytest.fixture(autouse=True, scope='class')
    def dtype(self):
        return torch.float

    @pytest.fixture(autouse=True, scope='class')
    def num_points(self):
        return 100000
    
    @pytest.fixture(autouse=True, scope='class')
    def yms(self):
        return 1e5
    
    @pytest.fixture(autouse=True, scope='class')
    def prs(self):
        return 0.45
    
    @pytest.fixture(autouse=True, scope='class')
    def rhos(self):
        return 100
    
    @pytest.fixture(autouse=True, scope='class')
    def appx_vol(self):
        return 1
    
    def sdBox(self, p):
        SDBOXSIZE = [1, 1, 1]
        b = np.array(SDBOXSIZE)
        q = np.absolute(p) - b
        return np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0], max(q[1], q[2])), 0.0)

    @pytest.fixture(autouse=True)
    def example_unit_cube_object(self, num_points, yms, prs, rhos, device, dtype):
        uniform_points = np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], size=(num_points, 3))
        sdf_vals = np.apply_along_axis(self.sdBox, 1, uniform_points)
        keep_points = np.nonzero(sdf_vals <= 0)[0]  # keep points where sd is not positive
        X0 = uniform_points[keep_points, :]
        X0_sdfval = sdf_vals[keep_points]

        YMs = np.full(X0.shape[0], yms)
        PRs = np.full(X0.shape[0], prs)
        Rhos = np.full(X0.shape[0], rhos)

        bb_vol = (np.max(uniform_points[:, 0]) - np.min(uniform_points[:, 0])) * (np.max(uniform_points[:, 1]) -
                                                                                np.min(uniform_points[:, 1])) * (np.max(uniform_points[:, 2]) - np.min(uniform_points[:, 2]))
        vol_per_sample = bb_vol / uniform_points.shape[0]
        appx_vol = vol_per_sample * X0.shape[0]
        return torch.tensor(X0, device=device, dtype=dtype), torch.tensor(X0_sdfval, device=device, dtype=dtype), torch.tensor(YMs, device=device, dtype=dtype), torch.tensor(PRs, device=device, dtype=dtype), torch.tensor(Rhos, device=device, dtype=dtype), torch.tensor(appx_vol, device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def example_rigid_cube(self, example_unit_cube_object, device, dtype):
        x0, x0_sdfval, yms, prs, rhos, appx_vol = example_unit_cube_object
        rigid_obj = SimplicitsObject.create_rigid(
            pts=x0, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)
        return rigid_obj

    @pytest.fixture(autouse=True)
    def scene_with_one_object(self, example_rigid_cube, device, dtype):
        scene = SimplicitsScene(device=device, dtype=dtype)
        scene.add_object(example_rigid_cube,
                         init_transform=torch.tensor([[1, 0, 0, 0],
                                                      [0, 1, 0, 10],
                                                      [0, 0, 1, 0]], device=device, dtype=dtype))
        return scene
    
    def test_simplicits_object_create_rigid(self, example_rigid_cube):
        rigid_obj = example_rigid_cube
        assert rigid_obj.num_handles == 1
        
    def test_simplicits_object_create_from_function(self, example_unit_cube_object, device, dtype):
        x0, x0_sdfval, yms, prs, rhos, appx_vol = example_unit_cube_object

        # Create object from skinning function that returns 2 weights for each point
        object_from_function = SimplicitsObject.create_from_function(
            pts=x0, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol, fcn=lambda x: torch.ones((x.shape[0], 2), device=device, dtype=dtype))
        assert object_from_function.num_handles == 2 

    def test_easy_api_scene_add_object_error_no_objects(self, example_rigid_cube, device, dtype):
        # Create
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        # Create scene
        scene = SimplicitsScene(device=device, dtype=dtype)

        # TODO: This will be fixed in another release
        # Set force without any objects in scene
        with pytest.raises(RuntimeError):
            scene.set_scene_gravity()
 
    def test_easy_api_scene_add_object_error_bad_init_transform(self, example_rigid_cube, device, dtype):
        # Create
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        # Create scene
        scene = SimplicitsScene(device=device, dtype=dtype)

        # Check Error: Add object to scene with incorrect init transform
        with pytest.raises(ValueError):
            scene.add_object(rigid_obj_one,
                             init_transform=torch.tensor([[1, 0, 0, 0],
                                                          [0, 1, 0, 0]], device=device, dtype=dtype))
        
    def test_easy_api_scene_add_object_error_after_sim_started(self, example_rigid_cube, device, dtype):
        # Create
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        # Create scene
        scene = SimplicitsScene(device=device, dtype=dtype)
        
       
        #Add objects to scene with correct init transforms
        scene.add_object(rigid_obj_one,
                            init_transform=torch.tensor([[1, 0, 0, 0],
                                                        [0, 1, 0, 10],
                                                        [0, 0, 1, 0]], device=device, dtype=dtype))
        
        scene.set_scene_gravity()
        
        # Check error: Add object to scene after simulation has started
        scene.run_sim_step()
        with pytest.raises(RuntimeError):
            scene.add_object(rigid_obj_two,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype))
 
    def test_easy_api_set_scene_gravity(self, scene_with_one_object):
        scene = scene_with_one_object
        scene.set_scene_gravity()
        assert scene.force_dict["pt_wise"]["gravity"] is not None 
        
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} is not within 1 unit of 10"


        for i in range(10):
            scene.run_sim_step()
            
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert mean_dyn_y < 10.0, f"Mean y coordinate of object {mean_dyn_y} has moved lower than 10 under gravity."
    
    def test_easy_api_set_scene_floor(self, scene_with_one_object):
        scene = scene_with_one_object
        scene.set_scene_gravity()
        scene.set_scene_floor(floor_height=5.0)
        assert scene.force_dict["pt_wise"]["floor"] is not None

        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} is not within 1 unit of 10"

        for i in range(50):
            scene.run_sim_step()

        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert mean_dyn_y < 10.0 and mean_dyn_y > 5.0, f"Mean y coordinate of object {mean_dyn_y} has moved lower than 10 under gravity, but greater than 5 under floor penalty."

    def test_easy_api_set_object_boundary_condition(self, scene_with_one_object):
        scene = scene_with_one_object
        scene.set_scene_gravity()
        # object is placed initally at y=10, so boundary condition is set to y>9.9
        scene.set_object_boundary_condition(0, "bdry_top", lambda x: x[:,1]>9.9) 
        
        assert scene.force_dict["pt_wise"]["bdry_top"] is not None
        assert scene.force_dict["pt_wise"]["bdry_top"]["object"].pinned_vertices.shape[0] == torch.sum(torch.where(scene.get_object_deformed_pts(0)[:,1]>9.9, 1, 0))
        pinned_indices = wp.to_torch(scene.force_dict["pt_wise"]["bdry_top"]["object"].pinned_indices)
        pinned_x = wp.to_torch(scene.force_dict["pt_wise"]["bdry_top"]["object"].pinned_vertices)
        check_allclose(scene.get_object_deformed_pts(0)[pinned_indices, 1], pinned_x[:, 1])
        
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} is not within 1 unit of 10"

        for i in range(20):
            scene.run_sim_step()

        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} should stay within 1 unit of 10 with fixed boundary"

    def test_easy_api_reset_scene(self, scene_with_one_object):
        scene = scene_with_one_object
        scene.set_scene_gravity()
        
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} is not within 1 unit of 10"

        assert scene.current_sim_step == 0
        
        
        for i in range(20):
            scene.run_sim_step()
            
        assert scene.current_sim_step == 20
            
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert mean_dyn_y < 10.0, f"Mean y coordinate of object {mean_dyn_y} has moved lower than 10 under gravity."
        
        scene.reset_scene()
        
        dyn_pts = scene.get_object_deformed_pts(0)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 10.0) < 1.0, f"Mean y coordinate of object {mean_dyn_y} is not within 1 unit of 10"
        
        assert scene.current_sim_step == 0
        
    def test_easy_api_enable_collisions_one_object(self, scene_with_one_object):
        scene = scene_with_one_object
        scene.set_scene_gravity()
        scene.set_scene_floor()
        scene.enable_collisions()
        
        assert scene.force_dict["collision"] is not None
        
        scene.run_sim_step()
    
    def test_easy_api_disabled_collisions(self, scene_with_one_object, example_rigid_cube, device, dtype):
        scene = scene_with_one_object
        scene.timestep = 0.05
        scene.newton_hessian_regularizer = 1e-4
        scene.add_object(example_rigid_cube,
                         init_transform=torch.tensor([[1, 0, 0, 0],
                                                      [0, 1, 0, 5],
                                                      [0, 0, 1, 0]], device=device, dtype=dtype))
        scene.set_scene_gravity()
        scene.set_scene_floor(floor_height=3.0)

        assert "collision" not in scene.force_dict
        # Check warning is raised when collision are not enabled with multiple objects
        with pytest.warns(UserWarning):
            collided_during_sim = False
            for i in range(50):
                scene.run_sim_step()

                if (scene.get_object_deformed_pts(0)[:, 1].mean()
                        - scene.get_object_deformed_pts(1)[:, 1].mean()) < 0.5:
                    collided_during_sim = True
                    break
            assert collided_during_sim, "Objects did not collide during simulation even though collisions are disabled"
        
    def test_easy_api_enabled_collisions(self, scene_with_one_object, example_rigid_cube, device, dtype):
        scene = scene_with_one_object
        scene.timestep = 0.03
        scene.newton_hessian_regularizer = 1e-5
        scene.add_object(example_rigid_cube,
                         init_transform=torch.tensor([[1, 0, 0, 0],
                                                      [0, 1, 0, 5],
                                                      [0, 0, 1, 0]], device=device, dtype=dtype))
        scene.set_scene_gravity()
        scene.set_scene_floor(floor_height=3.0)
        
        scene.reset_scene()
        
        scene.enable_collisions(
            impenetrable_barrier_ratio=0.5, collision_penalty=100.0)

        assert scene.force_dict["collision"] is not None

        for i in range(50):
            scene.run_sim_step()
            print(scene.get_object_deformed_pts(0)[:, 1].mean(
            ), scene.get_object_deformed_pts(1)[:, 1].mean())
            assert (scene.get_object_deformed_pts(0)[:, 1].mean() 
                    - scene.get_object_deformed_pts(1)[:, 1].mean()) > 0.5, \
            f"Mean y coordinate of object 0 is within 1 unit of object 1, so objects overlap at timestep {i}"
            
        
    def test_easy_api_scene_get_object_transform(self, example_rigid_cube, device, dtype):
        # Create
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        # Create scene
        scene = SimplicitsScene(device=device, dtype=dtype)
        # Add object to scene
        scene.add_object(rigid_obj_one,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    # move object 10 units in y direction
                                                    [0, 1, 0, 10],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype))
        scene.add_object(rigid_obj_two,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype), is_kinematic=True)

        # Assert after setup that mean y coordinate is 10
        obj_1_tfms = scene.get_object_transforms(0)
        expected_obj_1_tfms = torch.tensor([[0, 0, 0, 0],
                                            [0, 0, 0, 10],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 1]], device=device, dtype=dtype)
        assert torch.allclose(
            obj_1_tfms, expected_obj_1_tfms), f"First object is not at 0, 10, 0"

        obj_2_tfms = scene.get_object_transforms(1)
        expected_obj_2_tfms = torch.tensor([[0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 1]], device=device, dtype=dtype)
        assert torch.allclose(
            obj_2_tfms, expected_obj_2_tfms), f"Second object is not at 0, 0, 0"

    def test_easy_api_scene_set_object_initial_transform(self, example_rigid_cube, device, dtype):
        # Create
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        # Create scene
        scene = SimplicitsScene(device=device, dtype=dtype)
        # Add object to scene
        scene.add_object(rigid_obj_one,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 10], # move object 10 units in y direction
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype))
        scene.add_object(rigid_obj_two,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype), is_kinematic=True)

        scene.set_scene_gravity()
        scene.set_scene_floor()

        # Assert after setup that mean y coordinate is 10
        pts = scene.get_object_deformed_pts(0)
        expected_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
        assert torch.allclose(
            pts, expected_pts), f"Object is not at 0, 10, 0"
        
        # Check error: Set initial transform of object incorrectly with bad transform
        with pytest.raises(ValueError):
            scene.set_object_initial_transform(0, torch.tensor([[1, 0, 0, 0], # bad init transform
                                                                [0, 1, 0, 0]], device=device, dtype=dtype))
        
        # Check error: Set initial transform of kinematic object
        with pytest.raises(ValueError):
            scene.set_object_initial_transform(1, torch.tensor([[1, 0, 0, 0],
                                                                [0, 1, 0, 0],
                                                                [0, 0, 1, 0],
                                                                [0, 0, 0, 1]], device=device, dtype=dtype))
        
        # Check error: Set initial transform after simulation has started
        scene.run_sim_step()
        with pytest.raises(ValueError):
            scene.set_object_initial_transform(0, torch.tensor([[1, 0, 0, 0],
                                                                [0, 1, 0, 0],
                                                                [0, 0, 1, 0],
                                                                [0, 0, 0, 1]], device=device, dtype=dtype))
        
        
        # Set initial transform of object correctly
        scene.reset_scene()
        scene.set_object_initial_transform(0, torch.tensor([[1, 0, 0, 0],
                                                            [0, 1, 0, 0],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1]], device=device, dtype=dtype))
        
        pts = scene.get_object_deformed_pts(0)
        expected_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 0, 0], device=device, dtype=dtype).unsqueeze(0)
        assert torch.allclose(
            pts, expected_pts), f"Object is not set at 0, 0, 0"
        
    def test_easy_api_scene_setup_kinematic(self, example_rigid_cube, device, dtype):
        # 1. set kinematic object to be at 0, 10, 0 
        # 2. assert after setup that mean y coordinate is 10, store difference in y coordinate from dynamic object
        # 3. run simulation
        # 4. assert after a few steps that mean y coordinate is 10, assert difference in y coordinate from dynamic object is larger
        # 5. set kinematic object to be at 0, 5, 0
        # 6. assert after setup that mean y coordinate is 5
        # 7. run simulation
        # 8. assert after a few steps that mean y coordinate is 5, assert difference in y coordinate from dynamic object is smaller
        
        
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube
        
        scene = SimplicitsScene(device=device, dtype=dtype)
        scene.timestep = 0.1
        scene.add_object(rigid_obj_one, 
                        init_transform=torch.tensor([[1, 0, 0, 0], 
                                        [0, 1, 0, 10], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]], device=device, dtype=dtype),
                            is_kinematic=True)
        scene.add_object(rigid_obj_two,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 5],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], device=device, dtype=dtype))
        
        scene.set_scene_gravity()
        scene.set_scene_floor()
        
        # 2. assert after setup that mean y coordinate is 10
        kin_pts = scene.get_object_deformed_pts(0)
        expected_kin_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(
            kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
        
        dyn_pts=scene.get_object_deformed_pts(1)
        mean_dyn_y=dyn_pts[:, 1].mean()
        mean_kin_y = kin_pts[:, 1].mean()
        assert abs(mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5" 
        
        diff_in_y = abs(mean_kin_y - mean_dyn_y) # should be 5, but store this to measure later
        
        # 3. run simulation
        for i in range(10):
            scene.run_sim_step()
            
        # 4. assert that mean y coordinate of kinematic object is 10
        kin_pts = scene.get_object_deformed_pts(0)
        expected_kin_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(
            kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
        
        dyn_pts=scene.get_object_deformed_pts(1)
        mean_dyn_y=dyn_pts[:, 1].mean()
        mean_kin_y = kin_pts[:, 1].mean()
        assert abs(mean_kin_y - mean_dyn_y) > diff_in_y, f"Kinematic object is at y = {mean_kin_y}, dynamic object is at y = {mean_dyn_y}. The difference in y should be larger than {diff_in_y}. Either both objects are moving, or the dynamic object is not moving."
        
            
        # 5. Move kinematic object to 0, 1, 0 
        scene.set_kinematic_object_transform(0, 
                                            torch.tensor([[1,0,0,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], device=device, dtype=dtype))
        
        # 6. assert after that mean y coordinate of kinematic object is 1
        kin_pts = scene.get_object_deformed_pts(0)
        expected_kin_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 1, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(
            kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 1, 0"
        
        dyn_pts = scene.get_object_deformed_pts(1)
        mean_dyn_y = dyn_pts[:, 1].mean()
        mean_kin_y = kin_pts[:, 1].mean()
        assert abs(mean_kin_y - mean_dyn_y) < diff_in_y, f"Difference in y coordinate of dynamic object {mean_dyn_y} is not smaller than {diff_in_y} after moving kinematic object. Kinematic object was not moved."
        
    def test_easy_api_scene_reset_kinematic(self, example_rigid_cube, device, dtype):
        # 1. set dyn and kinematic objects to be at 0, 10, 0 and 0, 5, 0
        # 2. assert placement is correct
        # 3. run simulation
        # 4. assert locations 
        # 5. set kinematic object to be at 0, 1, 0
        # 5. reset scene
        # 6. assert dynamic object is at 0, 5, 0, kinematic object is at 0, 1, 0
    
        rigid_obj_one = example_rigid_cube
        rigid_obj_two = example_rigid_cube

        scene = SimplicitsScene(device=device, dtype=dtype)
        scene.timestep = 0.1
        scene.add_object(rigid_obj_one,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 10],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype),
                        is_kinematic=True)
        scene.add_object(rigid_obj_two,
                        init_transform=torch.tensor([[1, 0, 0, 0],
                                                    [0, 1, 0, 5],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], device=device, dtype=dtype))

        scene.set_scene_gravity()
        scene.set_scene_floor()

        # 2. assert after setup that mean y coordinate is 10
        kin_pts = scene.get_object_deformed_pts(0)
        mean_kin_y = kin_pts[:, 1].mean()
        expected_kin_pts = scene.get_object(0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(kin_pts, expected_kin_pts)
        
        dyn_pts = scene.get_object_deformed_pts(1)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5"

        # should be ~5, but store this to measure later
        diff_in_y = abs(mean_kin_y - mean_dyn_y)

        # 3. run simulation
        for i in range(10):
            scene.run_sim_step()

        # 4. assert that mean y coordinate of kinematic object is 10
        kin_pts = scene.get_object_deformed_pts(0)
        expected_kin_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(
            kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
        dyn_pts = scene.get_object_deformed_pts(1)
        mean_dyn_y = dyn_pts[:, 1].mean()
        mean_kin_y = kin_pts[:, 1].mean()
        assert abs(
            mean_kin_y - mean_dyn_y) > diff_in_y, f"Kinematic object is at y = {mean_kin_y}, dynamic object is at y = {mean_dyn_y}. The difference in y should be larger than {diff_in_y}. "

        # 5. Move kinematic object to 0, 1, 0
        scene.set_kinematic_object_transform(0,torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=dtype))

        # 6. Reset scene
        scene.reset_scene()
        
        # 7. assert dynamic object is at 0, 5, 0, kinematic object is at 0, 1, 0
        kin_pts = scene.get_object_deformed_pts(0)
        expected_kin_pts = scene.get_object(
            0).sample_pts + torch.tensor([0, 1, 0], device=device, dtype=dtype).unsqueeze(0)
        check_allclose(kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 1, 0"
        dyn_pts = scene.get_object_deformed_pts(1)
        mean_dyn_y = dyn_pts[:, 1].mean()
        assert abs(
            mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5"