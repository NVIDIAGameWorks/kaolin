from annotated_types import Ge, Le
import asyncio
import base64
import copy
from dataclasses import dataclass
import enum
import io
import logging
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from typing import Annotated, Dict, Any, Union, Callable

from humanfriendly.terminal import message

import kaolin.render.camera
import kaolin.visualize.web.io
from kaolin.visualize.web.sockets import AsyncMessageHandlerProtocol, AnyRendererMessageHandler, \
    GlobalWebSocketConnectionManager

from .input_abstraction import CloudAbstraction
from .sam import default_sam2_components, SimplerSAMSegmenter
from .segment import DisjointSegmentation
from .selection import CloudSelection, SelectAction
from .segmenter import GaussianSplatSegmenter
from .tracking import TrackerState, TrackingSession
from . import views
from .util import default_camera, mask_from_message


logger = logging.getLogger(__name__)


@dataclass
class ServerSideUserSettings:
    """ Defines per-client settings that persist on the server side, even if tab is reloaded."""
    max_tracking_resolution: Annotated[int, Ge(200), Le(1500)] = 900


class ServerApplicationState:
    """ In our app design, there is a shared scene state for the scene and its segmentation.
    In principle, multiple clients can modify it at the same time and get broadcast updates.
    """
    def __init__(self, cloud: CloudAbstraction,
                 cameras,
                 segmentation: DisjointSegmentation = None):
        """Original cloud representation."""
        self._orig_cloud = copy.deepcopy(cloud)

        """Any existing cameras for the scene."""
        self._cameras = cameras

        if segmentation is None:
            segmentation = DisjointSegmentation(len(cloud), cloud.device)
        self.__disjoint_segmentation = segmentation
        """Handles disjoint nature of the segments."""

    def fresh_cloud(self):
        return copy.deepcopy(self._orig_cloud)  # we don't allow mutating state cloud

    @property
    def cameras(self):
        return self._cameras

    @property
    def segmentation(self):
        # TODO: make segmentation object thread-safe
        return self.__disjoint_segmentation

class MessageTags(str, enum.Enum):
    """ Message tags our handler accepts or sends back. """

    # Incoming
    RENDER = 'render'  # render with current camera
    SET_CAMERA = 'set_camera'  # set current camera
    SAM_SEGMENT = 'sam_segment'  # perform and send back SAM result
    PROJECT = 'mask_project'  # project cam to 3D, with different actions (ADD, INTERSECT...)
    AGGREGATE = 'aggregate'  # track/aggregate or get info about that
    SEGMENT = 'segment'  # add/delete/edit global server-side segmentation state
    SEGMENT_REFRESH = 'segment_refresh'

    # Outgoing (not exhaustive; AnyRendererMessageHandler sends others)
    SET_SEGMENTS = 'set_segments'
    DONE_WITH_MASK = 'done_with_mask'
    DEBUG_TRACKING_VIEWS = 'debug_tracking_views'


class SegmentAction(str, enum.Enum):
    """ Actions one can take on the segments (not selections); see also SelectAction"""
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"
    SHOW = "show"
    SELECT = "select"


# TODO(Clement): implement these actions on the server side
class AggregateAction(str, enum.Enum):
    """ Actions one can take on aggregation-based tracking. """
    ADD_MASK = "add_mask"
    CLEAR_MASKS = "clear_masks"
    AGGREGATE = "aggregate"  # tracks and aggregates
    ADD_MASK_AND_AGGREGATE = "add_mask_and_aggregate"  # adds mask then immediately aggregates in one round-trip
    CLEAR_SELECTION = "clear_selection"  # empties the 3D point selection
    GET_DEBUG = "get_debug"  # should return num tracked masks, tracked cameras, and (if requested) specific tracked mask


def _encode_debug_images(tracking_outputs, masks_2d, max_dim=320):
    if not tracking_outputs:
        return {'count': 0, 'renders': [], 'masks': []}
    renders_b64 = []
    masks_b64 = []
    for out, mask in zip(tracking_outputs, masks_2d):
        mask = mask.squeeze()  # normalize (1, H, W) → (H, W); no-op if already (H, W)
        render = out['render']  # (3, H, W) float32 in [0, 1]
        _, H, W = render.shape
        scale = min(max_dim / H, max_dim / W, 1.0)
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))

        render_uint8 = (render.clamp(0, 1) * 255).to(torch.uint8)
        if scale < 1.0:
            render_uint8 = TF.resize(render_uint8, [new_h, new_w], antialias=True)
        buf = io.BytesIO()
        Image.fromarray(render_uint8.permute(1, 2, 0).cpu().numpy()).save(buf, format='JPEG', quality=70)
        renders_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

        mask_uint8 = (mask.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0)  # (1, H, W)
        if scale < 1.0:
            mask_uint8 = TF.resize(mask_uint8, [new_h, new_w], antialias=False,
                                    interpolation=InterpolationMode.NEAREST)
        buf = io.BytesIO()
        Image.fromarray(mask_uint8.squeeze(0).cpu().numpy(), mode='L').save(buf, format='PNG')
        masks_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    return {'count': len(renders_b64), 'renders': renders_b64, 'masks': masks_b64}


class InteractiveCloudSelector(AsyncMessageHandlerProtocol):
    """This class is the master Selection / Segmentation controller for an abstract representation of a monolithic scene.
       It handles transient selections, as well as disjoint segments for a scene on the server side.
       It has **no** knowledge of UI / interactions.

       Lifecycle: this handler is instantiated for each client and persists throughout the client session.
       """
    def __init__(self, state: ServerApplicationState,
                 settings: ServerSideUserSettings, render_kwargs=None):
        """
        Lifecycle: this handler is instantiated for each client and persists throughout the client session.

        Args:
            state:
            camera:
            settings:
            render_kwargs:
        """
        self.shared_state = state
        self.cloud = state.fresh_cloud()
        self.user_cloud = self.cloud  # includes visibility and annotations
        self._segment_visibility = {k: True for k in self.disjoint_segmentation.segments.keys()}

        num_points = len(self.cloud)
        device = self.cloud.device

        self.render_handler = AnyRendererMessageHandler(self.render_for_display, **render_kwargs)

        self._camera = default_camera().to(device)
        """Information needed to render gaussians on demand."""

        self.selection = CloudSelection.from_empty(num_points, device)
        """Handles 3D subset of points currently selected."""

        sam2_processor, sam2_model = default_sam2_components(device)
        self.sam_segmenter = SimplerSAMSegmenter(sam2_processor, sam2_model) if sam2_model is not None else None
        """Handles generation of SAM masks, and manages currently set view and points"""

        self._settings = settings
        """Maximum dimension enforced for the tracker rendered views. """

        self.tracker = TrackerState()
        """Accumulates (image, mask, camera) pairs per client for aggregation."""

    def render_for_display(self, camera):
        render_colors = self.user_cloud.render(camera)['render']  # TODO: use render passes here too
        render_colors = (render_colors.clip(0, 1) * 255).to(torch.uint8)

        # no alpha, we'll encode as jpeg
        #render_colors = torch.cat([render_colors, torch.ones_like(render_colors[..., :1]) * 255], dim=-1)
        return render_colors.squeeze(0)

    @property
    def device(self):
        return self.cloud.device

    SAM_RESULT_TAG = 'sam_result'
    """Outgoing message tag carrying the latest SAM mask back to the client."""

    def on_connection_open(self, write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        # Let's write dummy segments
        # dummy_segments = {'segments': [{'name': 'Segment 2001', 'visibe': True, 'can_delete': True, 'info': '5531'},
        #                                {'name': 'Beautiful flower vase', 'visible': False, 'can_delete': False, 'info': '66668'},
        #                                {'name': 'Amazing Aquarium', 'visible': False, 'can_delete': False, 'info': '66668'},
        #                                {'name': 'Whatever it is', 'visible': False, 'can_delete': False, 'info': '66668'},
        #                                {'name': 'Uni', 'visible': False, 'can_delete': False, 'info': '66668'},
        #                                {'name': 'Hedgehog', 'visible': False, 'can_delete': False, 'info': '66668'},
        #                                {'name': 'Uni', 'visible': False, 'can_delete': False, 'info': '66668'}]}
        # encoded = kaolin.visualize.web.io.encode_message(
        #     MessageTags.SET_SEGMENTS, dummy_segments, binary=True)
        # write_message_fn(encoded, True)
        encoded = kaolin.visualize.web.io.encode_message(
            MessageTags.SET_SEGMENTS, self.segment_info(), binary=True)
        write_message_fn(encoded, True)

    def accepted_message_tags(self) -> list[str]:
        return [
            MessageTags.SET_CAMERA,
            MessageTags.SAM_SEGMENT,
            MessageTags.PROJECT,
            MessageTags.AGGREGATE,
            MessageTags.SEGMENT
        ] + self.render_handler.accepted_message_tags()

    async def on_message(self, message_tag: str, message_content: Dict,
                         write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        # Use default handler for rendering
        if message_tag in self.render_handler.accepted_message_tags():
            await self.render_handler.on_message(message_tag, message_content, write_message_fn)

        # We handle those same and maybe additional requests too
        if message_tag == MessageTags.SET_CAMERA:
            logger.debug(f'InteractiveCloudSelector: Set camera '
                         f'(max_tracking_resolution={self._settings.max_tracking_resolution})')
            self._camera = kaolin.render.camera.Camera.from_dict(message_content).to(self.device)
        elif message_tag == MessageTags.SAM_SEGMENT:
            if self.sam_segmenter is None:
                logger.warning('Received sam_segment but no SAM segmenter is configured; ignoring.')
                return
            positions = message_content['positions']
            positions = torch.stack(positions) if len(positions) > 0 else torch.zeros((0, 2))
            labels = message_content['labels']
            # Offload rendering + SAM inference to a worker thread so we don't block the IO loop.
            await asyncio.to_thread(self._execute_sam_task, positions, labels, write_message_fn)
        elif message_tag == MessageTags.PROJECT:
            mask = message_content['mask']
            mask = mask_from_message(mask).to(self.device) > 20  # convert to bool
            selection_action = message_content['action']
            # After the 3D selection is updated, the cloud used for rendering is refreshed
            await asyncio.to_thread(self._execute_mask_project_task, selection_action, mask, write_message_fn)
            # TODO: also send some selection stats
        elif message_tag == MessageTags.AGGREGATE:
            await asyncio.to_thread(self._execute_aggregate_task, message_content, write_message_fn)
        elif message_tag == MessageTags.SEGMENT:
            name = message_content['name']
            segment_action = message_content['action']
            await asyncio.to_thread(self._execute_segment_task, name, segment_action, message_content, write_message_fn)
        elif message_tag == MessageTags.SEGMENT_REFRESH:
            raise NotImplementedError(f'Implement segment refresh from other clients!')
        elif message_tag not in self.render_handler.accepted_message_tags():
            logger.info(f'Unknown message tag {message_tag}')
            return

    def reset_selection(self):
        self.selection.reset()

    def project_2d_mask(self, mode: SelectAction, mask):
        transformed_xyz = self.cloud.project_means_to_2d(self._camera)
        new_selection = CloudSelection.from_image_mask(transformed_xyz, mask)
        self._apply_new_selection(new_selection, mode)

    @property
    def disjoint_segmentation(self):
        return self.shared_state.segmentation

    def segment_info(self):
        seg_meta = self.disjoint_segmentation.meta_as_dict()
        for seg in seg_meta['segments']:
            seg['visible'] = self._segment_visibility[seg['name']]
        return seg_meta

    def num_segments(self, enabled_only=False):
        if not enabled_only:
            return len(self.disjoint_segmentation.segments)
        return sum([1 for k, v in self.disjoint_segmentation.segments.items() if self._segment_visibility.get(k)])

    def enabled_segments_mask(self):
        mask = torch.zeros_like(self.disjoint_segmentation.get_background_segment().mask)
        for name, is_visible in self._segment_visibility.items():
            if is_visible:
                mask |= self.disjoint_segmentation.get_segment(name).mask
        return mask

    def _apply_new_selection(self, new_selection, mode: SelectAction):
        # Only allow selecting what is visible
        if self.some_segments_disabled():
            new_selection.intersect(
                CloudSelection.from_point_mask(self.enabled_segments_mask()))
        self.selection.apply(new_selection, mode)
        self._update_displayed_cloud()

    def _update_enabled_mask(self):
        """ Hide not enabled gaussians."""
        visible_mask = self.enabled_segments_mask()
        self._apply_new_selection(CloudSelection.from_point_mask(~visible_mask), SelectAction.REMOVE)
        self._update_displayed_cloud()

    def _update_displayed_cloud(self):
        visible_mask = self.enabled_segments_mask()
        self.user_cloud = self.cloud.with_selection_highlighted(self.selection.mask).with_visibility(visible_mask)

    def some_segments_disabled(self) -> True:
        return self.num_segments() != self.num_segments(enabled_only=True)

    def _add_segment(self, name):
        if self.selection.mask.sum() == 0:
            return False
        mask = copy.deepcopy(self.selection.mask)
        new_name = self.disjoint_segmentation.add_segment(name, mask)
        self._segment_visibility[new_name] = True
        self._update_enabled_mask()
        return True

    def _update_segment_mask(self, name):
        if self.selection.mask.sum() == 0:
            return False
        mask = copy.deepcopy(self.selection.mask)
        success = self.disjoint_segmentation.update_segment_mask(name, mask)
        self._update_enabled_mask()
        return success

    def _delete_segment(self, name):
        success = self.disjoint_segmentation.delete_segment(name)
        if success:
            del self._segment_visibility[name]
        self._update_enabled_mask()
        return success

    def _update_segment_name(self, name, target_name):
        if target_name == name:
            return False

        new_name = self.disjoint_segmentation.update_segment_name(name, target_name)
        if new_name is None:
            return False

        seg_visibility = self._segment_visibility[name]
        del self._segment_visibility[name]
        self._segment_visibility[new_name] = seg_visibility
        return True

    def _set_segment_visible(self, name, value):
        if name not in self._segment_visibility:
            logger.warning(f'Segment {name} DNE')
            return False

        self._segment_visibility[name] = value
        self._update_enabled_mask()
        return True

    def _set_segment_selected(self, name, selected):
        visibility = self._segment_visibility.get(name, None)
        segment = self.disjoint_segmentation.get_segment(name)
        if visibility is None:
            logger.warning(f'Segment {name} visibility DNE')
            return False
        if segment is None:
            logger.warning(f'Segment {name} DNE')
            return False

        action = SelectAction.ADD if selected else SelectAction.REMOVE
        self._apply_new_selection(CloudSelection.from_point_mask(segment.mask), action)
        return True

    def _execute_aggregate_task(self, message_content, write_message_fn):
        action = message_content.get('action')
        if action in (AggregateAction.ADD_MASK, AggregateAction.ADD_MASK_AND_AGGREGATE):
            mask = mask_from_message(message_content['mask']).to(self.device).float() / 255.0  # (H, W) float [0,1]
            # Clamp to max_tracking_resolution so Cutie pixel-memory dimensions are consistent
            # with the tracking renders (which are also clamped below).
            width, height = kaolin.render.camera.dimensions_to_max_resolution(
                self._camera.width, self._camera.height, self._settings.max_tracking_resolution)
            ref_camera = copy.deepcopy(self._camera)
            ref_camera.width = width
            ref_camera.height = height
            image = self.cloud.render(ref_camera, ['render'])['render'][0, ...].clip(0, 1).permute(2, 0, 1).float()  # (3, H, W)
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = TF.resize(mask.unsqueeze(0), [height, width],
                                 interpolation=InterpolationMode.NEAREST).squeeze(0)
            self.tracker.add_ground_truth_mask(image, mask, ref_camera)
            print(f'[AGGREGATE] add_mask: tracker now has {self.tracker.num_reference_masks} mask(s)')
            if action == AggregateAction.ADD_MASK_AND_AGGREGATE:
                action = AggregateAction.AGGREGATE  # fall through to aggregation below
            else:
                return
        if action == AggregateAction.AGGREGATE:
            if not self.tracker.can_track():
                print('[AGGREGATE] no reference masks; use Add Mask first')
                return
            print(f'[AGGREGATE] running tracking with {self.tracker.num_reference_masks} reference(s)')
            segmenter = GaussianSplatSegmenter(self.cloud.gsmodel)
            training_cameras = self.shared_state.cameras
            if training_cameras:
                # Resize to max_tracking_resolution so renders match reference image resolution.
                tracking_cameras = []
                for cam in training_cameras:
                    w, h = kaolin.render.camera.dimensions_to_max_resolution(
                        cam.width, cam.height, self._settings.max_tracking_resolution)
                    resized = copy.deepcopy(cam)
                    resized.width = w
                    resized.height = h
                    tracking_cameras.append(resized)
                print(f'[AGGREGATE] using {len(tracking_cameras)} training cameras')
            else:
                tracking_cameras = views.turnaround_cameras_from_camera_and_2dmask(
                    self.tracker.ref_mask, self.tracker.ref_cam, 24, segmenter)
                print(f'[AGGREGATE] no training cameras, using {len(tracking_cameras)} turnaround cameras')
            session = TrackingSession.from_tracker_state(self.tracker, device=str(self.device))
            result = session.track_and_segment(segmenter, tracking_cameras)
            mask_3d = result['3d_mask']
            print(f'[AGGREGATE] done: {mask_3d.sum().item()} / {len(mask_3d)} points selected')
            self._apply_new_selection(CloudSelection.from_point_mask(mask_3d), SelectAction.NEW)
            self.render_handler.render_and_send(write_message_fn)
            debug_payload = _encode_debug_images(result['outputs'], result['masks_2d'])
            encoded_debug = kaolin.visualize.web.io.encode_message(
                MessageTags.DEBUG_TRACKING_VIEWS, debug_payload, binary=True)
            write_message_fn(encoded_debug, True)
        elif action == AggregateAction.CLEAR_MASKS:
            self.tracker.reset_ground_truth_masks()
            print('[AGGREGATE] clear_masks: tracker reset')
        elif action == AggregateAction.CLEAR_SELECTION:
            self.reset_selection()
            self._update_displayed_cloud()
            self.render_handler.render_and_send(write_message_fn)
            print('[AGGREGATE] clear_selection: 3D selection emptied')
        else:
            print(f'[AGGREGATE] unhandled action={action}')

    def _execute_mask_project_task(self, selection_action, mask, write_message_fn):
        self.project_2d_mask(SelectAction(selection_action), mask)
        self.render_handler.render_and_send(write_message_fn)
        encoded = kaolin.visualize.web.io.encode_message(
            MessageTags.DONE_WITH_MASK, {}, binary=False)
        write_message_fn(encoded, False)

    def _execute_segment_task(self, name, action, message, write_message_fn):
        """Runs async action on the current segments, such as adding/deleting/visibility."""
        needs_fresh_render = True
        needs_seg_broadcast = True
        success = False
        if action == SegmentAction.ADD:
            success = self._add_segment(name)
        elif action == SegmentAction.DELETE:
            success = self._delete_segment(name)
        elif action == SegmentAction.UPDATE:
            target_name = message.get('new_name', None)
            if target_name is not None:
                success = self._update_segment_name(name, target_name)

            target_mask = message.get('new_mask', False)
            if target_mask:
                success = success or self._update_segment_mask(name)
        elif action in [SegmentAction.SHOW, SegmentAction.SELECT]:
            value = message.get('value', None)
            if value is None:
                logger.warning(f'For segment action {action}, "value" message field not set')
            elif action == SegmentAction.SHOW:
                success = self._set_segment_visible(name, value)
            elif action == SegmentAction.SELECT:
                success = self._set_segment_selected(name, value)
            needs_seg_broadcast = False
        else:
            logger.error(f'Unknown segment action {action}')

        if not success:
            logger.warning(f'Failed to perform segment action {message}')
            return

        # Send segment update back to this client's UI
        encoded = kaolin.visualize.web.io.encode_message(
            MessageTags.SET_SEGMENTS, self.segment_info(), binary=True)
        write_message_fn(encoded, True)

        # If needed, send fresh render
        if needs_fresh_render:
            self.render_handler.render_and_send(write_message_fn)

        if needs_seg_broadcast:
            GlobalWebSocketConnectionManager.instance().broadcast(encoded)


    def _execute_sam_task(self, positions, labels, write_message_fn):
        """Run rendering + SAM inference and send the resulting mask back.

        Runs synchronously inside an ``asyncio.to_thread`` worker.
        """
        width, height = kaolin.render.camera.dimensions_to_max_resolution(
            self._camera.width, self._camera.height, self._settings.max_tracking_resolution)
        sam_camera = copy.deepcopy(self._camera)
        sam_camera.width = width
        sam_camera.height = height

        if len(positions) == 0:
            # Empty point set: surface as an empty mask so the client clears any prior overlay.
            mask_uint8 = torch.zeros((height, width), dtype=torch.uint8)
        else:
            try:
                sam_cloud = self.cloud.with_visibility(self.enabled_segments_mask())
                view = (sam_cloud.render(sam_camera, ['render'])['render'][0, ...].clip(0, 1) * 255).to(torch.uint8)
                points = positions * torch.tensor([[width, height]], device=positions.device, dtype=positions.dtype)
                points = points.unsqueeze(0).unsqueeze(0).to(torch.int32).tolist()
                labels = labels.unsqueeze(0).unsqueeze(0).to(torch.int32).tolist()
                mask = self.sam_segmenter.compute_mask(view, points, labels)
                mask_uint8 = mask.to(torch.uint8) * 255
            except Exception as e:
                logger.exception(f'SAM compute_mask failed: {e}')
                return

        # We will encode as RGBA (larger), so that client can quickly render to canvas with opacity
        m = torch.cat([torch.ones_like(mask_uint8).unsqueeze(0).repeat(3, 1, 1) * 255, mask_uint8.unsqueeze(0)], dim=0)
        encoded = kaolin.visualize.web.io.encode_message(
            self.SAM_RESULT_TAG, {'img': m}, binary=True, image_format='png')
        write_message_fn(encoded, True)

