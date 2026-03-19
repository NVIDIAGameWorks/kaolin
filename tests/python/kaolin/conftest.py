import pytest
import torch
import kaolin
from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene
from kaolin.utils.testing import with_seed
from pathlib import Path


@pytest.fixture
def device():
    return "cuda"

@pytest.fixture
def dtype():
    return torch.float32

@pytest.fixture
def cantilever_beam_mesh():
    mesh_file_path = Path(__file__).resolve().parent / "physics" / "simplicits" / "regression_test_data" / "beam_surf.obj"
    mesh = kaolin.io.import_mesh(str(mesh_file_path), triangulate=True).cuda()
    return mesh

@pytest.fixture
def cube_drop_mesh():
    mesh_file_path = Path(__file__).resolve().parent / "physics" / "simplicits" / "regression_test_data" / "cube_surf.obj"
    mesh = kaolin.io.import_mesh(str(mesh_file_path), triangulate=True).cuda()
    return mesh


@pytest.fixture 
@with_seed(0,0,0)
def cantilever_beam_object(device, dtype, cantilever_beam_mesh):
    mesh = cantilever_beam_mesh
    num_samples = 100000

    uniform_pts = torch.rand(num_samples, 3, device=device) * (mesh.vertices.max(
        dim=0).values - mesh.vertices.min(dim=0).values) + mesh.vertices.min(dim=0).values

    boolean_signs = kaolin.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]  # m
    yms = 1e5*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m/s^2
    prs = 0.45*torch.ones(pts.shape[0], device=device, dtype=dtype)  # unitless
    rhos = 500*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m^3
    object_vol = (mesh.vertices.max(dim=0)[
                  0] - mesh.vertices.min(dim=0)[0]).prod()  # m^3 #bbx volume
    dt = 0.05  # s

    # get simplicits object
    weights_file_path = Path(__file__).resolve().parent / "physics" / "simplicits" / "regression_test_data" / "beam_weights_fcn_32_handles.pth"
    weights = torch.load(str(weights_file_path), weights_only=False)
    
    simplicits_object = SimplicitsObject.create_from_function(
        pts, yms, prs, rhos, object_vol, weights)
    return simplicits_object 

@pytest.fixture
@with_seed(0,0,0)
def cube_drop_object(device, dtype, cube_drop_mesh):
    mesh = cube_drop_mesh
    num_samples = 100000

    uniform_pts = torch.rand(num_samples, 3, device=device) * (mesh.vertices.max(
        dim=0).values - mesh.vertices.min(dim=0).values) + mesh.vertices.min(dim=0).values

    boolean_signs = kaolin.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]  # m
    yms = 1e4*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m/s^2
    prs = 0.45*torch.ones(pts.shape[0], device=device, dtype=dtype)  # unitless
    rhos = 500*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m^3
    object_vol = (mesh.vertices.max(dim=0)[
                  0] - mesh.vertices.min(dim=0)[0]).prod()  # m^3 #bbx volume

    # get simplicits object
    weights_file_path = Path(__file__).resolve().parent / "physics" / "simplicits" / "regression_test_data" / "cube_weights_fcn_32_handles.pth"
    weights = torch.load(str(weights_file_path), weights_only=False)
    simplicits_object = SimplicitsObject.create_from_function(
        pts, yms, prs, rhos, object_vol, weights)
    return simplicits_object


@pytest.fixture
def cantilever_beam_setup(cantilever_beam_object):
    """Fixture to set up cantilever beam scene for testing."""
    simplicits_object = cantilever_beam_object
    device = simplicits_object.device
    dtype = simplicits_object.dtype
    dt = 0.05

    scene = SimplicitsScene(
        device=device,
        dtype=dtype,
        timestep=dt,
        max_newton_steps=10,  # run to near convergence
        max_ls_steps=20,
    )
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True
    
    scene.add_object(simplicits_object, num_qp=1024)

    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)

    scene.set_object_boundary_condition(
        0, "right", lambda x: x[:, 0] >= 0.98, bdry_penalty=10000.0)

    return scene


@pytest.fixture
@with_seed(0, 0, 0)
def cube_drop_setup(cube_drop_object):
    """Fixture to set up cube drop scene for testing."""
    simplicits_object = cube_drop_object
    device = simplicits_object.device
    dtype = simplicits_object.dtype
    dt = 0.05

    scene = SimplicitsScene(
        device=device,
        dtype=dtype,
        timestep=dt,
        max_newton_steps=10,  # run to near convergence
        max_ls_steps=20,
    )
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True

    scene.add_object(simplicits_object, num_qp=1024)

    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)

    return scene
