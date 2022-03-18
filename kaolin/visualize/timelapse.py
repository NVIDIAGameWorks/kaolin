import glob
import logging
import os
import re
import posixpath
import warnings

try:
    from pxr import Usd
except ImportError:
    warnings.warn("Warning: module pxr not found", ImportWarning)

from kaolin import io

logger = logging.getLogger(__name__)

__all__ = [
    'Timelapse',
    'TimelapseParser'
]

class Timelapse:
    def __init__(self, log_dir, up_axis='Y'):
        self.logdir = log_dir

    def _add_shading_variant(self, prim, name):
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

    def add_pointcloud_batch(self, iteration=0, category='', pointcloud_list=None, colors=None, points_type="point_instancer"):
        """
        Add pointclouds to visualizer output.

        Args:
            iteration (int): Positive integer identifying the iteration the supplied pointclouds belong to.
            category (str, optional): Batch name.
            pointcloud_list (list of tensors, optional): Batch of point clouds as (B x N x 3) tensor or list of variable
               length point cloud tensors, each (N_i x 3).
            colors (list of tensors, optional): Batch of RGB colors of length N.
            points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer. 
                               "usd_geom_points" indicates UsdGeomPoints and "point_instancer" indicates PointInstancer. 
                               Please refer here for UsdGeomPoints:
                               https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
                               https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. 
                               Default: "point_instancer".
        """
        validated = self._validate_parameters(
            pointcloud_list=pointcloud_list, colors=colors,
        )
        pointcloud_list, colors = validated

        pc_path = posixpath.join(self.logdir, category)
        os.makedirs(pc_path, exist_ok=True)

        if colors is None:
            colors = [None] * len(pointcloud_list)

        for i, sample in enumerate(zip(pointcloud_list, colors)):
            points, colour = sample
            # Establish default USD file paths for sample
            pc_name = f'pointcloud_{i}'
            ind_out_path = posixpath.join(pc_path, f'{pc_name}.usd') 

            if not os.path.exists(ind_out_path):
                # If sample does not exist, create it.
                stage = io.usd.create_stage(ind_out_path)
                if points_type == "usd_geom_points":
                    stage.DefinePrim(f'/{pc_name}', 'Points')
                elif points_type == "point_instancer":
                    stage.DefinePrim(f'/{pc_name}', 'PointInstancer')
                else:
                    raise ValueError(f"Expected points_type to be 'usd_geom_points' or 'point_instancer', but got '{points_type}'.")
                stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{pc_name}'))
            else:
                stage = Usd.Stage.Open(ind_out_path)
            pc_prim = stage.GetPrimAtPath(f'/{pc_name}')

            # Adjust end timecode to match current iteration
            stage.SetEndTimeCode(iteration)
            io.usd.add_pointcloud(stage, points, f'/{pc_name}', colors=colour, time=iteration, points_type=points_type)

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

        vg_path = posixpath.join(self.logdir, category)
        os.makedirs(vg_path, exist_ok=True)

        for i, sample in enumerate(zip(voxelgrid_list, colors, semantic_ids)):
            voxelgrid, colour, semantic_id = sample
            # Establish default USD file paths for sample
            vg_name = f'voxelgrid_{i}'
            ind_out_path = posixpath.join(vg_path, f'{vg_name}.usd') 

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

        meshes_path = posixpath.join(self.logdir, category)
        textures_path = posixpath.join(meshes_path, 'textures')
        os.makedirs(meshes_path, exist_ok=True)
        os.makedirs(textures_path, exist_ok=True)

        for i, sample in enumerate(zip(*validated)):
            vertices, faces, uvs, face_uvs_idx, face_normals, materials = sample
            # Establish default USD file paths for sample
            mesh_name = f'mesh_{i}'
            ind_out_path = posixpath.join(meshes_path, f'{mesh_name}.usd') 

            if not os.path.exists(ind_out_path):
                # If sample does not exist, create it.
                stage = io.usd.create_stage(ind_out_path)
                mesh_prim = stage.DefinePrim(f'/{mesh_name}', 'Mesh')
                stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{mesh_name}'))
                if materials_list is not None:
                    for material_name, _ in materials_list[i].items():
                        self._add_shading_variant(mesh_prim, material_name)
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
                        material.write_to_usd(
                            ind_out_path, f'/{mesh_name}/{material_name}',
                            time=iteration,
                            texture_dir='textures',
                            texture_file_prefix=f'{mesh_name}_{material_name}_{iteration}_',
                            bound_prims=[mesh_prim])
            stage.Save()


def _get_timestamps(filenames):
    """
    Returns the timestamps of all filenames as a dictionary keyed by
    filename. Will throw error if files do not exits.
    """
    res = {}
    for f in filenames:
        res[f] = os.stat(f).st_mtime_ns
    return res


class TimelapseParser(object):
    """
    Utility class for working with log directories created using the Timelapse
    interface. For example, this class can be used to extract raw data from
    the written checkpoints as follows:

    Example:
        # During training
        timelapse = Timelapse(log_dir)
        timelapse.add_pointcloud_batch(iteration=idx, category="prediction", pointcloud_list=[predictions])

        # Examining training run
        parser = TimelapseParser(log_dir)
        path = parser.get_file_path("pointcloud", "prediction", 0)
        cloud = kaolin.io.usd.import_pointclouds(path, time=iteration_number)  # time should be iteration number
        # Visualize, save or analyze as desired
    """
    __SUPPORTED_TYPES = ["mesh", "pointcloud", "voxelgrid"]

    class CategoryInfo:
        """
        Corresponds to a "category" specified in Timelapse for a specific type
        like "mesh". The ids corresponds to the number of objects
        saved in calls like add_mesh_batch, and end_time is the largest end
        time in the group.
        """
        def __init__(self, category, ids=None, end_time=0):
            self.category = category
            self.ids = [] if ids is None else list(ids)
            self.end_time = end_time

        def serializable(self):
            return {'category': self.category,
                    'ids': self.ids,
                    'end_time': self.end_time}

        def __repr__(self):
            return repr((self.category, len(self.ids), self.end_time))

        def __lt__(self, other):
            return repr(self) < repr(other)

        def add_instance(self, new_id, end_timecode):
            if new_id in self.ids:
                raise RuntimeError('Id {} already added for category {}'.format(new_id, self.category))
            self.ids.append(new_id)
            self.ids.sort()
            self.end_time = max(self.end_time, end_timecode)

    def __init__(self, log_dir):
        self.logdir = log_dir

        # { (typestr, category, id) : path }
        self.filepaths = TimelapseParser.get_filepaths(self.logdir)
        self.timestamps = _get_timestamps(self.filepaths.values())

        # { typestr : [CategoryInfo (serializable)] }
        self.dir_info = TimelapseParser.parse_filepath_info(self.filepaths)
        logger.debug(self.dir_info)

    def get_file_path(self, type, category, id):
        """Gets file path by keys.
        Args:
            type (str): one of "mesh", "pointcloud", "voxelgrid"
            category (str): category passed to Timelapse when writing checkpoints
            id (int): id of the item within its batch
        Return:
            (str) or None
        """
        fpath_key = (type, category, int(id))
        if fpath_key not in self.filepaths:
            logger.error('Key {} not in filepaths: {}'.format(fpath_key, self.filepaths))
            return None
        return self.filepaths[fpath_key]

    def check_for_updates(self):
        """Updates parse information if it has changed in the logdir.

        Returns:
            (bool) - True if updates exist, False if not
        """
        filepaths = TimelapseParser.get_filepaths(self.logdir)
        timestamps = _get_timestamps(filepaths.values())
        if timestamps != self.timestamps:
            logger.info('Changes to logdirectory detected: {}'.format(self.logdir))
            self.filepaths = filepaths
            self.timestamps = timestamps
            self.dir_info = TimelapseParser.parse_filepath_info(self.filepaths)
            return True
        return False

    @staticmethod
    def _count_items(cat_infos):
        total = 0
        for cat in cat_infos:
            total += len(cat['ids'])
        return total

    def num_mesh_items(self):
        return TimelapseParser._count_items(self.dir_info['mesh'])

    def num_pointcloud_items(self):
        return TimelapseParser._count_items(self.dir_info['pointcloud'])

    def num_voxelgrid_items(self):
        return TimelapseParser._count_items(self.dir_info['voxelgrid'])

    def num_mesh_categories(self):
        return len(self.dir_info['mesh'])

    def num_pointcloud_categories(self):
        return len(self.dir_info['pointcloud'])

    def num_voxelgrid_categories(self):
        return len(self.dir_info['voxelgrid'])

    def get_category_names_by_type(self, type):
        if type not in self.dir_info:
            return [x['category'] for x in self.dir_info[type]]

    def get_category_info(self, type, category):
        if type not in self.dir_info:
            return None
        return next((x for x in self.dir_info[type] if x['category'] == category), None)

    @staticmethod
    def get_filepaths(logdir):
        """Get all USD file paths within a directory that match naming conventions imposed by Timelapse.

        Args:
            logdir (str): root directory where USD timelapse files are written

        Returns:
            dict: keyed by tuples (typestr, category, id_within_batch) with values
             containing full USD file paths
        """
        fname_pattern = '(.*)_([0-9]+).usd'

        filepaths = {}
        for typestr in TimelapseParser.__SUPPORTED_TYPES:
            usd_pattern = '{}*.usd'.format(typestr)
            files = glob.glob(os.path.join(logdir, '**', usd_pattern), recursive=True)

            if len(files) == 0:
                logger.info('No checkpoints found for type {}: no files matched pattern {} in {}'.format(
                    typestr, usd_pattern, logdir))

            for fpath in files:
                cat = os.path.dirname(os.path.relpath(fpath, logdir))
                m = re.match(fname_pattern, os.path.basename(fpath))
                if m is None:
                    logger.error('USD {} basename does not match pattern {}'.format(
                        fpath, fname_pattern))
                    continue
                num = int(m.group(2))
                filepaths[(typestr, cat, num)] = fpath

        return filepaths

    @staticmethod
    def parse_filepath_info(filepaths):
        """Parses a directory of checkpoints written by Timelapse module into a summary format.

        Args:
            filepaths: dictionary output by get_filepaths

        Returns:
            dictionary keyed by checkpoint type ("mesh", "pointcloud", "voxelgrid")
                      with each value a list of (serializable) CategoryInfo
        """
        info = {}  # { "mesh" : { "category": CategoryInfo} }
        for k, fpath in filepaths.items():
            stage = Usd.Stage.Open(fpath)

            # Note: stage.GetEndTimeCode() may store incorrect value
            times = io.usd.get_authored_time_samples(fpath)
            if len(times) == 0:
                end_time = 0
            else:
                end_time = times[-1]

            typestr, cat, id_num = k

            if typestr not in info:
                info[typestr] = {}
            if cat not in info[typestr]:
                info[typestr][cat] = TimelapseParser.CategoryInfo(cat)
            info[typestr][cat].add_instance(id_num, end_time)

        result = {}  # { "mesh": [ CategoryInfo (serializable dict) ] }
        for typestr, catdict in info.items():
            result[typestr] = [x.serializable() for x in sorted(catdict.values())]

        for typestr in TimelapseParser.__SUPPORTED_TYPES:
            if typestr not in result:
                result[typestr] = []   # Ensure all types are represented

        return result
