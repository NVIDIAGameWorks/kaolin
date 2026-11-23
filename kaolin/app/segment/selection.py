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

from __future__ import annotations
from abc import ABC, abstractmethod
import enum
import logging
import torch

import kaolin.utils.testing
from kaolin.utils.log import log_tensor

logger = logging.getLogger(__name__)


def regenerate_identifier():
    return torch.randint(0, 2**32, (1,)).to(torch.int32)


class SelectAction(str, enum.Enum):
    NEW = "new"
    ADD = "add"
    REMOVE = "subtract"
    INTERSECT = "intersect"


class Selection(ABC):
    """ Represents a selection, agnostic to the type of elements.
    """

    def __init__(self):
        self.identifier: torch.IntTensor = regenerate_identifier()
        """ Helps track status changes to the selection, this number is randomized with every action over the
            selection. Querying it allows to quickly check if selection "almost probably" changed, with a high chance.  
        """

    def apply(self, other, action: SelectAction):
        if action == SelectAction.NEW:
            return self.select(other)
        elif action == SelectAction.ADD:
            return self.add(other)
        elif action == SelectAction.REMOVE:
            return self.remove(other)
        elif action == SelectAction.INTERSECT:
            return self.intersect(other)
        else:
            raise RuntimeError(f'Unknown action {action}')

    @abstractmethod
    def select(self, other: Selection):
        """ Perform a new selection from scratch, essentially using the selected elements from "other" """
        raise NotImplementedError

    @abstractmethod
    def add(self, other: Selection):
        """ Adds elements from other selection to current selection """
        raise NotImplementedError

    @abstractmethod
    def remove(self, other: Selection):
        """ Removes elements from current selection, if they appear in the other selection """
        raise NotImplementedError

    @abstractmethod
    def intersect(self, other: Selection):
        """ Intersects the current selection with another selection.
        Only elements which appear in both will be kept.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Clear selection. """
        raise NotImplementedError

    @abstractmethod
    def select_all(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class DenseSelection(Selection):
    """ Represents a dense selection of anything with a mask. E.g, mask can be points, or it can be a 2D
    array for an image. User is responsible for ensuring selections have compatible masks.
    """
    def __init__(self, mask: torch.BoolTensor):
        """
        Constructs a DenseSelection object which keeps a reference to the mask. Changing mask values will also update
        this selection. Clone the tensor before passing it in if this behavior is not desirable.
        """
        super().__init__()
        assert mask is not None

        self.mask: torch.BoolTensor = mask
        """ Holds for each shape element whether it's selected or not """

    @torch.no_grad()
    def count(self):
        return self.mask.sum().item()

    @torch.no_grad()
    def select(self, other: DenseSelection):
        """ Perform a new selection from scratch, essentially using the selection mask from "other" """
        # Override current selection mask with other selection
        self.mask = other.mask
        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def add(self, other: DenseSelection):
        """ Adds elements from other selection to current selection """
        if self.mask is not None:
            self.mask |= other.mask
        else:
            self.mask = other.mask

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def remove(self, other: DenseSelection):
        """ Removes elements from current selection, if they appear in the other selection """
        if self.mask is not None:
            self.mask &= ~other.mask
        else:
            self.mask = other.mask & False

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def intersect(self, other: DenseSelection):
        """ Intersects the current selection with another selection.
        Only elements which appear in both will be kept.
        """
        if self.mask is not None:
            self.mask &= other.mask
        else:
            self.mask = other.mask & False

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    def reset(self):
        """ Clear selection. """
        self.mask[:] = False

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    def select_all(self):
        self.mask[:] = True

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    def set_mask(self, mask):
        """ Sets the mask in the DenseSelection object to a reference of the mask provided. Changing mask  values will
        also update this selection. Clone the input tensor before passing it in if this behavior is not desirable.
        """
        self.mask = mask

        # Update identifier since selection changed
        self.identifier = regenerate_identifier()
        return self

    def __len__(self):
        """ Number of elements selected. """
        return 0 if self.mask is None else torch.sum(self.mask)


class CloudSelection(DenseSelection):
    """
    Dense selection over splatted points, providing extra constructors.
    """
    def __init__(self, mask):
        super().__init__(mask)

    @staticmethod
    def from_empty(num_points, device) -> CloudSelection:
        return CloudSelection(torch.zeros((num_points,), device=device).to(torch.bool))

    @staticmethod
    def from_point_mask(mask) -> CloudSelection:
        """
        Creates selection given mask active over points.
        """
        return CloudSelection(mask)

    @staticmethod
    def from_image_mask(splatted_points, image_mask) -> CloudSelection:
        """
        Takes points that have already been splatted (or projected) onto the image plane
        and selects those that fall into the True regions of the provided image_mask.

        Args:
            splatted_points: N x 3 tensor of points projected onto image plane; assumes -1..1 NDC range;
                e.g. see project_gaussian_means_to_2d
            image_mask: H x W bool tensor
        """
        mask = CloudSelection._mask_from_image(splatted_points, image_mask)
        return CloudSelection(mask)

    @staticmethod
    def from_image_mask_within_depth(splatted_points, image_mask, depth_image, threshold=0.1):
        """
        Takes points that have already been splatted (or projected) onto the image plane
        and selects those that fall into the True regions of the provided image_mask AND
        whose splatted depth (third channel) is within threshold of the depth
        rendered into the depth_image at their location.

        Args:
            splatted_points: N x 3 tensor of points projected onto image plane; assumes -1..1 NDC range;
                e.g. see project_gaussian_means_to_2d; with third dimension having same scale as depth_image
            image_mask: H x W bool tensor
            depth_image: H x W float tensor
            threshold: float
        """
        mask = CloudSelection._mask_from_image(splatted_points, image_mask, depth_image, threshold=threshold)
        return CloudSelection(mask)

    @staticmethod
    def _mask_from_image(splatted_points, image_mask, depth_image=None, threshold=0.1):
        assert image_mask.dtype == torch.bool, f'unsupported mask {kaolin.utils.testing.tensor_info(image_mask, print_stats=True)}'
        assert len(image_mask.shape) == 2, f'{image_mask.shape}'

        dims = torch.tensor([image_mask.shape[1], image_mask.shape[0]]).to(splatted_points.dtype).to(
            splatted_points.device).reshape(-1, 2)
        coord_xy = torch.round((splatted_points[:, :2] + 1) * 0.5 * dims).to(torch.long)

        coord_invalid = (coord_xy[:, 0] < 0) & (coord_xy[:, 1] < 0) & \
                        (coord_xy[:, 0] >= image_mask.shape[1]) & (coord_xy[:, 1] >= image_mask.shape[0])

        coord_xy[:, 0] = coord_xy[:, 0].clamp(0, image_mask.shape[1] - 1)
        coord_xy[:, 1] = coord_xy[:, 1].clamp(0, image_mask.shape[0] - 1)
        coord_xy[:, 1] = image_mask.shape[0] - 1 - coord_xy[:, 1] # reverse y-coord
        coord_xy[:, 1] = coord_xy[:, 1].clamp(0, image_mask.shape[0] - 1)
        in_mask = image_mask[coord_xy[:, 1], coord_xy[:, 0]]
        mask = in_mask & ~coord_invalid

        if depth_image is None:
            return mask

        log_tensor(depth_image, 'depth_image', logger, print_stats=True)
        delta = depth_image[coord_xy[:, 1], coord_xy[:, 0]] - splatted_points[:, 2]
        depth_mask = (torch.abs(delta) < threshold)
        # TODO: make select more in front
        # (delta < 0) & (-delta < threshold * 3)))  # more selected in front
        return depth_mask & mask


    @staticmethod
    @torch.no_grad()
    def from_bounds(splatted_points, bounds) -> CloudSelection:
        """
        Takes points that have already been splatted (or projected) onto the image plane
        and selects those that fall into the normalized bounds.

        Args:
            splatted_points: N x 3 tensor of points projected onto image plane; assumes -1..1 NDC range;
                e.g. see project_gaussian_means_to_2d
            bounds: x0, y0, x1, y1 in the -1..1 range
        """
        x0, y0, x1, y1 = bounds
        mask = (splatted_points[:, 0] >= x0) & (splatted_points[:, 0] <= x1) & \
            (splatted_points[:, 1] >= y0) & (splatted_points[:, 1] <= y1)

        return CloudSelection(mask.to(torch.bool))
