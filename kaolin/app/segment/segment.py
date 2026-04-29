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

import logging
import torch

from kaolin.visualize.web.naming import UniqueIdGenerator

logger = logging.getLogger(__name__)

# TODO: this is very redundant; we should just store one int mask for all the segments
class Segment:
    """ Represents a segment of an object / scene."""
    id_generator = UniqueIdGenerator()

    def __init__(self, mask, name=None):
        # A boolean mask tensor specifying which of the objects elements belong to the segment
        # If the object changes, this mask should be updated or the object becomes invalid
        self.mask = mask

        if name is None:
            name = Segment.id_generator.get_unique_id_local('Segment')

        self.name = name

    @classmethod
    def from_dict(cls, dict_val):
        return Segment(mask=dict_val['mask'], name=dict_val['name'])

    def meta_as_dict(self):
        return {'name': self.name, 'info': f'{self.mask.sum().item()}'}

    def as_dict(self):
        res = self.meta_as_dict()
        res['mask'] = self.mask
        return res

    def is_empty(self) -> bool:
        """ True if the segment is empty, e.g. contains nothing """
        return not self.mask.any()


# TODO: this class needs to be thread-safe
class DisjointSegmentation:
    """
    Handles the logic of keeping track of N disjoint segments, ensuring this property.
    """
    BACKGROUND_NAME = "background"

    def __init__(self, num_points, device, create_segment=None, masks=None):
        """
        Constructs segmentation for a set of items ("points").

        Args:
            num_points: total number of points
            device: device to store things on
            create_segment: constructor(mask, name) for the segmentation.entities.Segment subclass, e.g. containing any extra info
            masks: optional existing masks
        """
        self.create_segment = create_segment if create_segment is not None else Segment
        self.num_points = num_points
        self.device = device
        self.segments = {DisjointSegmentation.BACKGROUND_NAME: self.__create_background_segment()}
        self.id_generator = UniqueIdGenerator()
        self.id_generator.reset_ids([DisjointSegmentation.BACKGROUND_NAME])

        if masks is not None:
            for name, mask in masks.items():
                self.add_segment(name)
                self.update_segment_mask(name, mask)

    @property
    def foreground_segments(self):
        return {k: v for k, v in self.segments.items() if k != DisjointSegmentation.BACKGROUND_NAME}

    def __create_background_segment(self, value=1):
        if value == 1:
            mask = torch.ones((self. num_points,), dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros((self. num_points,), dtype=torch.bool, device=self.device)
        return self.create_segment(mask, DisjointSegmentation.BACKGROUND_NAME)

    def meta_as_dict(self):
        seg_meta = [{**s.meta_as_dict(), 'can_delete': self.can_delete_segment(n)}  for n, s in self.segments.items()]
        return {'segments': seg_meta}

    def as_dict(self):
        seg = {n: s.as_dict() for n, s in self.segments.items()}
        return seg

    def legacy_save_segments_to_file(self, path):
        torch.save(self.as_dict(), path)

    def legacy_load_segments_from_file(self, path):
        segment_dict = torch.load(path)
        if len(segment_dict) == 0:
            return False
        return self.init_from_dict(segment_dict)

    def init_from_dict(self, segment_dict):
        num_pts = -1
        for k, v in segment_dict.items():
            if num_pts < 0:
                num_pts = v['mask'].shape[0]
            else:
                if v['mask'].shape[0] != num_pts:
                    logger.error(f'Saved segments have inconsistent mask sizes')
                    return False

        self.num_points = num_pts
        self.segments = {k: self.create_segment.from_dict(v) for k, v in segment_dict.items()}
        for k, s in self.segments.items():
            s.mask = s.mask.to(self.device)

        if DisjointSegmentation.BACKGROUND_NAME not in segment_dict:
            logger.warning(f'Loaded segments have no {DisjointSegmentation.BACKGROUND_NAME}, adding dummy one')
            self.segments[DisjointSegmentation.BACKGROUND_NAME] = self.__create_background_segment(0)
        return True

    def num_segments(self):
        return len(self.segments)

    def has_segment(self, name):
        return name in self.segments

    def get_segment(self, name) -> Segment:
        return self.segments.get(name)

    def get_background_segment(self):
        return self.get_segment(DisjointSegmentation.BACKGROUND_NAME)

    def _check_segment_exists(self, name, raise_error=True):
        exists = self.has_segment(name)
        if not exists:
            msg = f'Segment with name {name} not found; existing segments: {self.segments.keys()}'
            if raise_error:
                raise KeyError(msg)
            else:
                logger.warning(msg)
        return exists

    def num_segment_points(self, name):
        self._check_segment_exists(name)
        return torch.sum(self.segments[name].mask).item()

    def add_segment(self, name, mask=None):
        if self.has_segment(name):
            prev_name = name
            name = self.id_generator.get_unique_id(name)
            logger.warning(f'Segment {prev_name} already exists; using name {name}')

        self.segments[name] = self.create_segment(
            torch.zeros((self.num_points,), dtype=torch.bool, device=self.device), name)
        if mask is not None:
            self.update_segment_mask(name, mask)
        return name

    def update_segment_name(self, name, new_name):
        if not self._check_segment_exists(name, raise_error=False):
            return None

        if name == DisjointSegmentation.BACKGROUND_NAME:
            logger.warning(f'Cannot rename background segment.')
            return None

        segment = self.get_segment(name)
        self.delete_segment(name)
        new_name = self.create_segment(new_name, segment.mask)
        self.id_generator.reset_ids(self.segments.keys())
        return new_name

    def update_segment_mask(self, name, mask):
        """
        Updates mask for segment with name {name}. Removes True points from all other segments.
        """
        self._check_segment_exists(name)
        logger.info(f'Updating segment {name} with new mask')
        if name == DisjointSegmentation.BACKGROUND_NAME:
            logger.warning(f'Cannot manually set background mask.')
            return False

        removed_mask = self.segments[name].mask & ~mask
        for seg_name, seg in self.segments.items():
            prev_pts = self.num_segment_points(seg_name)
            if seg_name == name:
                self.segments[seg_name].mask = mask
            elif seg_name == DisjointSegmentation.BACKGROUND_NAME:
                self.segments[seg_name].mask[removed_mask] = True  # removed points go to background
                seg.mask[mask] = False
            else:
                seg.mask[mask] = False  # new mask trumps all existing masks
            logger.info(f'  > Updated segment {seg_name}: {prev_pts}pts --> {self.num_segment_points(seg_name)}')

        return True

    def prune_segments(self, prune_mask):
        valid_points_mask = ~prune_mask
        for seg_name, seg in self.segments.items():
            prev_pts = self.num_segment_points(seg_name)
            seg.mask = seg.mask[valid_points_mask]
            logger.info(f'  > Updated segment {seg_name}: {prev_pts}pts --> {self.num_segment_points(seg_name)}')

    def append_points(self, num_points, segment_name=BACKGROUND_NAME):
        for seg_name, seg in self.segments.items():
            if seg_name == segment_name:
                append_mask = torch.ones(num_points).to(seg.mask)
            else:
                append_mask = torch.zeros(num_points).to(seg.mask)
            seg.mask = torch.cat([seg.mask, append_mask], dim=0)

    def delete_segment(self, name):
        if not self.can_delete_segment(name):
            return False
        self.segments[DisjointSegmentation.BACKGROUND_NAME].mask |= self.segments[name].mask
        del self.segments[name]
        self.id_generator.reset_ids(self.segments.keys())
        return True

    def can_delete_segment(self, name):
        exists = self._check_segment_exists(name, raise_error=False)
        return exists and not name == DisjointSegmentation.BACKGROUND_NAME

