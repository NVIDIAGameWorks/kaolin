import os

from pxr import Usd, UsdShade
from kaolin import io


class Timelapse:
    def __init__(self, log_dir, up_axis='Y'):
        self.logdir = log_dir

    def _add_shading_variant(self, stage, prim, name):
        vset = prim.GetVariantSets().AddVariantSet("shadingVariant")
        vset.AddVariant(name)
        return vset

    def _validate_parameters(self, **kwargs):
        lengths = {k: len(v) for k, v in kwargs.items() if v is not None}
        assert lengths, 'No attributes provided.'
        num_samples = list(lengths.values())[0]
        assert all([v == num_samples for v in lengths.values()]), \
            f'Number of samples for each attribute must be equal: {lengths}'

        return_params = []
        for v in kwargs.values():
            if v is None:
                return_params.append([None] * num_samples)
            else:
                return_params.append(v)
        return return_params

    def add_pointcloud_batch(self, iteration=0, category='', pointcloud_list=None, colors=None, semantic_ids=None):
        """
        Add pointclouds to visualizer output.

        Args:
            iteration (int): Positive integer identifying the iteration the supplied pointclouds belong to.
            pointcloud_list (list of tensors, optional): Batch of points of length N defining N pointclouds.
            colors (list of tensors, optional): Batch of RGB colors of length N.
            semantic_ids (list of int, optional): Batch of semantic IDs.
        """
        validated = self._validate_parameters(
            pointcloud_list=pointcloud_list, colors=colors, semantic_ids=semantic_ids,
        )
        pointcloud_list, colors, semantic_ids = validated

        pc_path = os.path.join(self.logdir, category)
        os.makedirs(pc_path, exist_ok=True)

        for i, sample in enumerate(zip(pointcloud_list, colors, semantic_ids)):
            points, colour, semantic_id = sample
            # Establish default USD file paths for sample
            pc_name = f'pointcloud_{i}'
            ind_out_path = os.path.join(pc_path, f'{pc_name}.usd') 

            if not os.path.exists(ind_out_path):
                # If sample does not exist, create it.
                stage = io.usd.create_stage(ind_out_path)
                stage.DefinePrim(f'/{pc_name}', 'PointInstancer')
                stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{pc_name}'))
            else:
                stage = Usd.Stage.Open(ind_out_path)
            pc_prim = stage.GetPrimAtPath(f'/{pc_name}')

            # Adjust end timecode to match current iteration
            stage.SetEndTimeCode(iteration)

            # Set each attribute supplied
            if points is not None:
                io.usd.add_pointcloud(stage, points, f'/{pc_name}', time=iteration)
            if colour is not None:
                raise NotImplementedError
            if semantic_id is not None:
                raise NotImplementedError
            stage.Save()

    def add_voxelgrid_batch(self, iteration=0, category='', voxelgrid_list=None, colors=None, semantic_ids=None):
        """
        Add voxelgrids to visualizer output.

        Args:
            iteration (int): Positive integer identifying the iteration the supplied voxelgrids belong to.
            category (str, optional): Batch name.
            voxelgrid_list (list of tensors, optional): Batch of points of length N defining N pointclouds.
            colors (list of tensors, optional): Batch of RGB colors of length N.
            semantic_ids (list of int, optional): Batch of semantic IDs.
        """
        validated = self._validate_parameters(
            voxelgrid_list=voxelgrid_list, colors=colors, semantic_ids=semantic_ids,
        )
        voxelgrid_list, colors, semantic_ids = validated

        vg_path = os.path.join(self.logdir, category)
        os.makedirs(vg_path, exist_ok=True)

        for i, sample in enumerate(zip(voxelgrid_list, colors, semantic_ids)):
            voxelgrid, colour, semantic_id = sample
            # Establish default USD file paths for sample
            vg_name = f'voxelgrid_{i}'
            ind_out_path = os.path.join(vg_path, f'{vg_name}.usd') 

            if not os.path.exists(ind_out_path):
                # If sample does not exist, create it.
                stage = io.usd.create_stage(ind_out_path)
                stage.DefinePrim(f'/{vg_name}', 'PointInstancer')
                stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{vg_name}'))
            else:
                stage = Usd.Stage.Open(ind_out_path)
            vg_prim = stage.GetPrimAtPath(f'/{vg_name}')

            # Adjust end timecode to match current iteration
            stage.SetEndTimeCode(iteration + 1)     # HACK currently need one frame later to get the last iteration

            # Set each attribute supplied
            if voxelgrid is not None:
                io.usd.add_voxelgrid(stage, voxelgrid, f'/{vg_name}', time=iteration)
            if colour is not None:
                raise NotImplementedError
            if semantic_id is not None:
                raise NotImplementedError

            stage.Save()


    def add_mesh_batch(self, iteration=0, category='', vertices_list=None, faces_list=None,
                       uvs_list=None, face_uvs_idx_list=None, face_normals_list=None, materials_list=None):
        """
        Add meshes to visualizer output.

        Args:
            iteration (int): Positive integer identifying the iteration the supplied meshes belong to.
            category (str, optional): Batch name.
            vertices_list (list of tensors, optional): Vertices for N meshes of shape (num_vertices, 3).
            faces_list (list of tensors, optional): Faces for N meshes of shape (num_faces, face_size).
            uvs_list (list of tensors, optional): UV coordinates for N meshes of shape (num_uvs, 2).
            face_uvs_idx_list (list of tensors, optional): Index of UV coordinates for N meshes of shape (num_faces, face_size).
            face_normals_list (list of tensors, optional): Face normals for N meshes of shape (num_faces, face_size, 3).
            materials_list (list, optional): List of materials for N meshes. For each mesh, if a list of io.Materials is
                supplied, each material is applied to the mesh as a ShadingVariant. A name for each material can be defined
                by supplying a dictionary in the form of {'material_name': material}.
        """
        validated = self._validate_parameters(
            vertices_list=vertices_list, faces_list=faces_list, uvs_list=uvs_list,
            face_uvs_idx_list=face_uvs_idx_list, face_normals_list=face_normals_list,
            materials_list=materials_list
        )

        meshes_path = os.path.join(self.logdir, category)
        textures_path = os.path.join(meshes_path, 'textures')
        os.makedirs(meshes_path, exist_ok=True)
        os.makedirs(textures_path, exist_ok=True)

        for i, sample in enumerate(zip(*validated)):
            vertices, faces, uvs, face_uvs_idx, face_normals, materials = sample
            # Establish default USD file paths for sample
            mesh_name = f'mesh_{i}'
            ind_out_path = os.path.join(meshes_path, f'{mesh_name}.usd') 

            if not os.path.exists(ind_out_path):
                # If sample does not exist, create it.
                stage = io.usd.create_stage(ind_out_path)
                mesh_prim = stage.DefinePrim(f'/{mesh_name}', 'Mesh')
                stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{mesh_name}'))
                if materials_list is not None:
                    for material_name, _ in materials_list[i].items():
                        self._add_shading_variant(stage, mesh_prim, material_name)
            else:
                stage = Usd.Stage.Open(ind_out_path)
                mesh_prim = stage.GetPrimAtPath(f'/{mesh_name}')

            # Adjust end timecode to match current iteration
            stage.SetEndTimeCode(iteration)

            io.usd.add_mesh(stage, f'/{mesh_name}', vertices, faces, uvs,
                            face_uvs_idx, face_normals, time=iteration)
            if materials is not None:
                if isinstance(materials, io.materials.Material):
                    materials = {'0': materials}
                if isinstance(materials, list):
                    materials = {str(i): v for i, v in enumerate(materials)}
                vset = mesh_prim.GetVariantSets().GetVariantSet('shadingVariant')
                for material_name, material in materials.items():
                    vset.SetVariantSelection(material_name)
                    material.usd_root_path = meshes_path
                    with vset.GetVariantEditContext():
                        material_prim = material.write_to_usd(ind_out_path, f'/{mesh_name}/{material_name}', time=iteration,
                                                              texture_dir='textures',
                                                              texture_file_prefix=f'{mesh_name}_{material_name}_{iteration}_')
                        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
                        binding_api.Bind(UsdShade.Material(material_prim))

            stage.Save()
