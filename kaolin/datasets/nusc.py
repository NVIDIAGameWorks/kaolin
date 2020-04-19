# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
from functools import reduce

import torch
import numpy as np

try:
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
except ImportError as err:
    import_error = err
else:
    import_error = None

from .base import KaolinDataset
from kaolin.rep import PointCloud


class NuscDetection(KaolinDataset):
    r"""nuScenes dataloader https://www.nuscenes.org/
    Args:
        nusc (NuScenes): nuScenes data parser
        train (bool): If True, return the training set, otherwise the val set
        nsweeps (int): number of lidar sweeps ( 1 <= nsweeps)
        min_distance (float): minimum distance of lidar points to origin (in order to remove points on the ego car)
    Returns:
        .. code-block::

           dict: {
                attributes: {},
                data: {pc: kaolin.rep.PointCloud, boxes: list}
           }
    Example:
        >>> nusc = NuScenes(version='v1.0-mini', dataroot='../data/nuscenes')
        >>> traindata = kal.datasets.NuscDetection(nusc, train=True, nsweeps=5)
        >>> inst = traindata[0]
        >>> inst['data']['pc'].points.shape
        torch.Size([24630, 5])
    """

    def initialize(self, nusc: NuScenes, train: bool, nsweeps: int, min_distance: float = 2.2):
        assert(nsweeps >= 1), f'nsweeps {nsweeps} should be 1 or greater'
        if import_error is not None:
            raise import_error

        self.nusc = nusc
        self.train = train
        self.nsweeps = nsweeps
        self.min_distance = min_distance

        self.scenes = self._get_scenes()
        self.samples = self._get_samples()

    def _get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
            'lyft': {True: 'lyft_train', False: 'lyft_val'},
        }[self.nusc.version][self.train]

        return create_splits_scenes()[split]

    def _get_samples(self):
        samples = list(self.nusc.sample)

        # remove samples that aren't in this set
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.samples)

    def _get_data(self, index):
        sample = self.samples[index]

        # point cloud
        pc = get_lidar_data(self.nusc, sample, self.nsweeps,
                            min_distance=self.min_distance)
        pc = PointCloud(torch.from_numpy(pc.T))

        # bounding boxes
        boxes = get_boxes(self.nusc, sample)

        return {'pc': pc, 'boxes': boxes}

    def _get_attributes(self, index):
        return {}


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """Similar to LidarPointCloud.from_file_multisweep but returns in the ego car frame."""
    # Init.
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = 1e-6 * current_sd_rec['timestamp'] - ref_time
        times = np.full((1, current_pc.nbr_points()), time_lag)

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def get_boxes(nusc, rec):
    """Simplified version of nusc.get_boxes"""
    egopose = nusc.get('ego_pose',
                       nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    trans = -np.array(egopose['translation'])
    rot = Quaternion(egopose['rotation']).inverse
    boxes = []
    for tok in rec['anns']:
        inst = nusc.get('sample_annotation', tok)

        box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']),
                  name=inst['category_name'])
        box.translate(trans)
        box.rotate(rot)

        boxes.append(box)

    return boxes
