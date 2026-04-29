# TODO(Clement): build on or delete this utility we had in the past

import logging

logger = logging.getLogger(__name__)



class TrackerState:
    """
    Utility to keep currently set / tracked masks.
    """
    def __init__(self):
        self._ref_images = []
        """ Currently set reference images for selecting a given object."""

        self._ref_masks = []
        """ Currently set reference masks for that given object."""

        self._ref_cams = []
        """ Cameras for the currently set reference images and masks."""

        self._ref_has_occlusions = []
        """ Bool status on whether the mask has occlusions. """

        self._result = None
        self._tracked_cameras = None

    def can_track(self):
        return len(self._ref_images) > 0

    def update_rendered_views(self, cloud):
        pass

    def reset_ground_truth_masks(self):
        self._ref_images = []
        self._ref_masks = []
        self._ref_cams = []
        self._ref_has_occlusions = []

    def add_ground_truth_mask(self, image, mask, cam, has_occlusions=False):
        # TODO: resize if needed
        # TODO: figure out proper input conventions
        self._ref_images.append(image)
        self._ref_masks.append(mask)
        self._ref_cams.append(cam)
        self._ref_has_occlusions.append(has_occlusions)

    @property
    def num_reference_masks(self):
        return len(self._ref_images)

    @property
    def ref_image(self):
        return self._ref_images[0]

    @property
    def ref_mask(self):
        return self._ref_masks[0]

    @property
    def ref_cam(self):
        return self._ref_cams[0]

    @property
    def ref_cameras(self):
        return self._ref_cams

    # TODO: maybe some utilities to get or set tracked mask results?

    @property
    def num_predicted_masks(self):
        if self._result is None:
            return 0
        return len(self._result['2d_masks'])

    def predicted_mask(self, idx):
        if self._result is None:
            return None
        return self._result['2d_masks'][idx]

    def target_view(self, idx):
        if self._result is None:
            return None
        return self._result['outputs'][idx]['render']

    def target_cam(self, idx):
        if self._tracked_cameras is None:
            return None
        return self._tracked_cameras[idx]


