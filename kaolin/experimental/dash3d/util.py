# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
import json
import logging
import numpy as np

from pxr import Usd, UsdGeom
from tornado.websocket import WebSocketHandler
import tornado.gen

import kaolin.io.usd
from kaolin.visualize import TimelapseParser

logger = logging.getLogger(__name__)


def meshes_to_binary(vertices_list, faces_list):
    """Encodes meshes in a binary format for transferring over the network.

    Args:
        vertices_list: list of numpy array V x 3 (float32; will convert to this)
        faces_list: list of numpy array F x 3 (int32; will convert to this)

    Returns:
        bytes
    """
    nmeshes = len(vertices_list)
    if len(faces_list) != nmeshes:
        raise RuntimeError(
            'Expected equal number of vertex and face lists, got: {}, {}'.format(
                nmeshes, len(faces_list)))

    # TODO: if needed, specify consistent order in tobytes()
    texture_mode = 0  # TODO: use to extend support with backward compatibility
    tbd_info0 = 0
    tbd_info1 = 0
    binstr = np.array([nmeshes, texture_mode, tbd_info0, tbd_info1], dtype=np.int32).tobytes()

    for i in range(nmeshes):
        vertices = vertices_list[i]
        faces = faces_list[i]
        # TODO: better to assume we always reshape USD data consistently before calling this
        nvertices = vertices.size // 3
        nfaces = faces.size // 3
        binstr += np.array([nvertices, nfaces], dtype=np.int32).tobytes()
        # TODO: ideally stream raw USD chunk without even parsing
        binstr += vertices.astype(np.float32).tobytes()
        binstr += faces.astype(np.int32).tobytes()

    return binstr


def point_clouds_to_binary(positions_list):
    """Encodes point clouds in a binary format for transferring over the network.

    Args:
        positions_list (list of numpy arrays): P x 3 (float32; will convert to this)
    Returns:
        bytes
    """
    nclouds = len(positions_list)
    texture_mode = 0  # TODO: use to extend support with backward compatibility
    tbd_info0 = 0
    tbd_info1 = 0
    binstr = np.array([nclouds, texture_mode, tbd_info0, tbd_info1], dtype=np.int32).tobytes()

    for i in range(nclouds):
        positions = positions_list[i]
        # TODO: better to assume we always reshape USD data consistently before calling this
        npts = positions.size // 3
        binstr += np.array([npts, 0], dtype=np.int32).tobytes()
        # Also include bounding box, mins, then maxes
        binstr += np.min(positions, axis=0).astype(np.float32).tobytes()
        binstr += np.max(positions, axis=0).astype(np.float32).tobytes()
        # TODO: avoid going through numpy; ideally stream raw USD chunk without even parsing
        binstr += positions.astype(np.float32).tobytes()

    return binstr


class StreamingGeometryHelper(object):
    """Helper class for parsing and preparing geometry updates to the client.
    """
    def __init__(self, logdir):
        self.logdir = logdir
        self.parser = TimelapseParser(logdir)

    def get_directory_info(self):
        self.parser.check_for_updates()
        return self.parser.dir_info

    @staticmethod
    def _find_snap_time(brackets, target_time):
        """Returns the closest time to target_time out of bracketing time samples already provided by USD.
        Args:
            brackets (list of Number): list of length 2 of two time samples surrounding the target_time
            target_time (Number): target time to find a snap for
        Returns:
            (Number)
        """
        if abs(brackets[0] - target_time) < abs(brackets[1] - target_time):
            return brackets[0]
        return brackets[1]

    @staticmethod
    def _does_snap_time_require_update(snap_time, current_time):
        """
        Returns true if snap_time is sufficiently far from current_time to send geometry update.
        Args:
            snap_time (Number)
            current_time (Number)
        Returns:
            (bool)
        """
        if current_time is not None and abs(snap_time - current_time) < 0.5:
            logger.info('Snap time {} too close to current_time {}; no geometry parsed'.format(
                snap_time, current_time))
            return False
        return True

    @staticmethod
    def _pick_one_scene_path(fpath, scene_paths, type_str):
        if len(scene_paths) == 0:
            logger.warning('USD {} has no data of type {}'.format(fpath, type_str))
            return None
        if len(scene_paths) > 1:
            logger.warning('USD {} has more than one scene path of type {},'
                           'only one will be shown: {}'.format(fpath, type_str, scene_paths))
        return scene_paths[0]

    def parse_encode_pointcloud(self, category, id, target_time, current_time=None):
        """Retrieves pointcloud by its user-given category, id within batch and timecode.
        Geometry is only parsed and encoded if target_time is sufficiently far from current_time.
        Retrieved geometry is encoded in binary if an update is required.

        Note that this function *never* interpolates attributes in time, but snaps to
        the closest time to avoid misrepresenting the Timelapse data.

        Args:
            category (str):
            id (int):
            target_time (Number):
            current_time (Number):
        Return:
            (binary_str, Number) - encoded geometry or None, and snap_time found
        """
        fpath = self.parser.get_file_path("pointcloud", category, id)
        if fpath is None:
            return None, 0

        scene_path = self._pick_one_scene_path(
            fpath, kaolin.io.usd.get_pointcloud_scene_paths(fpath), "pointcloud")
        if scene_path is None:
            return None, 0

        stage = Usd.Stage.Open(fpath)
        time_brackets = kaolin.io.usd.get_pointcloud_bracketing_time_samples(stage, scene_path, target_time=target_time)
        snap_time = StreamingGeometryHelper._find_snap_time(time_brackets, target_time)
        if not StreamingGeometryHelper._does_snap_time_require_update(snap_time, current_time):
            return None, current_time

        # TODO: is there overhead to creating stage?
        points = kaolin.io.usd.import_pointcloud(fpath, scene_path, snap_time).points
        return point_clouds_to_binary([points.numpy()]), snap_time

    def parse_encode_mesh(self, category, id, target_time, current_time=None):
        fpath = self.parser.get_file_path("mesh", category, id)
        if fpath is None:
            return None, 0

        scene_path = self._pick_one_scene_path(
            fpath, kaolin.io.usd.get_scene_paths(fpath, prim_types=['Mesh']), "mesh")
        if scene_path is None:
            return None, 0

        # TODO: modify io.usd API to be usable here instead once we add support for textures
        stage = Usd.Stage.Open(fpath)
        mesh_prim = stage.GetPrimAtPath(scene_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        if mesh is None:
            return None, 0

        # May only need to update vertices, not faces
        vertices = None
        triangles = None
        time_brackets_verts = mesh.GetPointsAttr().GetBracketingTimeSamples(target_time)
        time_brackets_faces = mesh.GetFaceVertexIndicesAttr().GetBracketingTimeSamples(target_time)

        snap_time_verts = StreamingGeometryHelper._find_snap_time(time_brackets_verts, target_time)
        snap_time_faces = StreamingGeometryHelper._find_snap_time(time_brackets_faces, target_time)

        if StreamingGeometryHelper._does_snap_time_require_update(snap_time_verts, current_time):
            snap_time = snap_time_verts
            vertices = np.array(mesh.GetPointsAttr().Get(time=snap_time_verts),
                                dtype=np.float32)

        # TODO: support streaming update to only one of values
        # if StreamingGeometryHelper._does_snap_time_require_update(snap_time_faces, current_time):
        if vertices is not None:
            triangles = np.array(mesh.GetFaceVertexIndicesAttr().Get(time=snap_time_faces),
                                 dtype=np.int32)

        if triangles is None and vertices is None:
            return None, current_time

        return meshes_to_binary([vertices], [triangles]), snap_time


class GeometryWebSocketHandler(WebSocketHandler):
    """ Handles websocket communication with the JS client.
    """

    def initialize(self, helper):
        """ Takes existing StreamingGeometryHelper as input."""
        # Note: this is correct, __init__ method should not be written for this
        self.helper = helper

    def open(self):
        """ Open socket connection and send information about available geometry."""
        logger.debug("Socket opened.")
        message = {"type": "dirinfo",
                   "data": self.helper.get_directory_info()}
        self.write_message(message, binary=False)

    @tornado.gen.coroutine
    def on_message(self, message):
        """ Handles new messages on the socket."""
        logger.debug("Received message of type {}: {}".format(type(message), message))

        try:
            msg = json.loads(message)
        except Exception as e:
            logger.error('Failed to decode incoming message: {}'.format(e))
            return

        if msg.get("type") == "geometry":
            requests = msg.get("data")
            if not requests:
                logger.error('Message contained no data: {}'.format(msg))

            logger.debug('Handling geometry request: {}'.format(msg))

            for req in requests:
                byte_message = yield self.get_requested_geometry(req)
                self.maybe_write_message(byte_message, binary=True)

    @tornado.gen.coroutine
    def get_requested_geometry(self, req):
        """Finds, parses and encodes requested geometry if update is required;
        returns None of geometry can't be found or no update is required (i.e. requested
        timestamp is equal to the current timestamp on the client).

        Args:
            req (dict): parsed client side json request with the following fields
                type (str): required geometry type in "mesh", "pointcloud"
                category (str): the user-provided category when writing the Timelapse
                id (convertible to int): the id within the batch written by Timelapse
                time (convertible to float): the timestamp of the requested geometry
                view_id (int): id of the client side view to send result to
                current_time (convertible to float): time of the currently displayed
                    geometry on the client side, used to skip unneeded updates (optional)
        Returns:
            None or binary string encoding:
                - first 32bit int: type_id with 0 for mesh, 1 for pointcloud
                - 2nd  32bit int: view_id from the request
                - 3rd 32bit int: snap_time or the actual timestamp of the returned geometry
                - 4th 32bit int: reserved for extra info in the future
                - remaining bytes: encoding of the geometry
        """
        byte_geometry = None
        snap_time = 0
        type_id = 0

        required_attr = ["type", "category", "id", "time", "view_id"]
        for att in required_attr:
            if att not in req:
                logger.error("Request missing key {}: {}".format(att, req))
                return None

        req_id = int(req["id"])
        req_time = float(req["time"])
        req_current_time = float(req.get("current_time")) if "current_time" in req else None

        if req.get("type") == "mesh":
            type_id = 0
            try:
                byte_geometry, snap_time = self.helper.parse_encode_mesh(
                    req["category"], req_id, req_time, current_time=req_current_time)
            except Exception as e:
                logger.error('Failed to obtain mesh for request {}: {}'.format(req, e))
        elif req.get("type") == "pointcloud":
            type_id = 1
            try:
                byte_geometry, snap_time = self.helper.parse_encode_pointcloud(
                    req["category"], req_id, req_time, current_time=req_current_time)
            except Exception as e:
                logger.error('Failed to obtain pointcloud for request {}: {}'.format(req, e))
        else:
            logger.error('Unsupported requested geometry type: {}'.format(req.get("type")))

        if byte_geometry is None:
            return None

        bininfo = np.array([type_id, req["view_id"], snap_time, 0], dtype=np.int32).tobytes()  # TODO: order
        return bininfo + byte_geometry

    @tornado.gen.coroutine
    def maybe_write_message(self, byte_message, binary=True):
        """ Writes message, unless it is None."""
        if byte_message is not None:
            self.write_message(byte_message, binary=binary)

    def on_close(self):
        logger.info("Socket closed.")
