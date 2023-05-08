# ==============================================================================================================
# The following snippet shows how to use kaolin to preprocess a shapenet dataset
# To quickly sample point clouds from the mesh at runtime
# ==============================================================================================================
# See also:
#  - Documentation: ShapeNet dataset
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.shapenet.html#kaolin.io.shapenet.ShapeNetV2
#  - Documentation: CachedDataset
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.dataset.html#kaolin.io.dataset.CachedDataset
#  - Documentation: Mesh Ops:
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.mesh.html
#  - Documentation: Obj loading:
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html
# ==============================================================================================================

import argparse
import os
import torch

import kaolin as kal

parser = argparse.ArgumentParser(description='')
parser.add_argument('--shapenet-dir', type=str, default=os.getenv('KAOLIN_TEST_SHAPENETV2_PATH'),
                    help='Path to shapenet (v2)')
parser.add_argument('--cache-dir', type=str, default='/tmp/dir',
                    help='Path where output of the dataset is cached')
parser.add_argument('--num-samples', type=int, default=10,
                    help='Number of points to sample on the mesh')
parser.add_argument('--cache-at-runtime', action='store_true',
                    help='run the preprocessing lazily')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers during preprocessing (not used with --cache-at-runtime)')

args = parser.parse_args()


def preprocessing_transform(inputs):
    """This the transform used in shapenet dataset __getitem__.

    Three tasks are done:
    1) Get the areas of each faces, so it can be used to sample points
    2) Get a proper list of RGB diffuse map
    3) Get the material associated to each face
    """
    mesh = inputs['mesh']
    vertices = mesh.vertices.unsqueeze(0)
    faces = mesh.faces
    
    # Some materials don't contain an RGB texture map, so we are considering the single value
    # to be a single pixel texture map (1, 3, 1, 1)
    # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
    uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0) % 1, (0, 0, 0, 1)) * 2. - 1.
    uvs[:, :, 1] = -uvs[:, :, 1]
    face_uvs_idx = mesh.face_uvs_idx
    face_material_idx = mesh.material_assignments
    materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).float() / 255. if 'map_Kd' in m else
                 m['Kd'].reshape(1, 3, 1, 1)
                 for m in mesh.materials]

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0
    face_uvs = kal.ops.mesh.index_vertices_by_faces(
        uvs, face_uvs_idx
    )
    face_uvs[:, mask] = 0.

    outputs = {
        'vertices': vertices,
        'faces': faces,
        'face_areas': kal.ops.mesh.face_areas(vertices, faces),
        'face_uvs': face_uvs,
        'materials': materials,
        'face_material_idx': face_material_idx,
        'name': inputs['name']
    }

    return outputs

class SamplePointsTransform(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        coords, face_idx, feature_uvs = kal.ops.mesh.sample_points(
            inputs['vertices'],
            inputs['faces'],
            num_samples=self.num_samples,
            areas=inputs['face_areas'],
            face_features=inputs['face_uvs']
        )
        coords = coords.squeeze(0)
        face_idx = face_idx.squeeze(0)
        feature_uvs = feature_uvs.squeeze(0)

        # Interpolate the RGB values from the texture map
        point_materials_idx = inputs['face_material_idx'][face_idx]
        all_point_colors = torch.zeros((self.num_samples, 3))
        for i, material in enumerate(inputs['materials']):
            mask = point_materials_idx == i
            point_color = torch.nn.functional.grid_sample(
                material,
                feature_uvs[mask].reshape(1, 1, -1, 2),
                mode='bilinear',
                align_corners=False,
                padding_mode='border')
            all_point_colors[mask] = point_color[0, :, 0, :].permute(1, 0)

        outputs = {
            'coords': coords,
            'face_idx': face_idx,
            'colors': all_point_colors,
            'name': inputs['name']
        }
        return outputs

# Make ShapeNet dataset with preprocessing transform
ds = kal.io.shapenet.ShapeNetV2(root=args.shapenet_dir,
                                categories=['dishwasher'],
                                train=True,
                                split=0.1,
                                with_materials=True,
                                output_dict=True,
                                transform=preprocessing_transform)

# Cache the result of the preprocessing transform
# and apply the sampling at runtime
pc_ds = kal.io.dataset.CachedDataset(ds,
                                     cache_dir=args.cache_dir,
                                     save_on_disk=True,
                                     num_workers=args.num_workers,
                                     transform=SamplePointsTransform(args.num_samples),
                                     cache_at_runtime=args.cache_at_runtime,
                                     force_overwrite=True)


for data in pc_ds:
    print("coords:\n", data['coords'])
    print("face_idx:\n", data['face_idx'])
    print("colors:\n", data['colors'])
    print("name:\n", data['name'])
