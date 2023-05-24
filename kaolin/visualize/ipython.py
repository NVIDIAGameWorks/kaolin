# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

from abc import abstractmethod
from collections.abc import Sequence
from io import BytesIO
import math
import traceback
import warnings

from PIL import Image as PILImage
import torch

try:
    from ipyevents import Event
    from ipywidgets import Output
    from ipywidgets import Image as ImageWidget
    from ipycanvas import Canvas
except Exception as e:
    warnings.warn('Error importing kaolin.visualize.ipython:\n' + traceback.format_exc())


from ..render.camera import CameraExtrinsics

__all__ = [
    'update_canvas',
    'BaseIpyVisualizer',
    'IpyTurntableVisualizer',
    'IpyFirstPersonVisualizer',
]

def update_canvas(canvas, image):
    assert isinstance(image, torch.Tensor) and image.dtype == torch.uint8, \
           "image must be a torch.Tensor of uint8 "
    assert isinstance(canvas, Canvas)
    f = BytesIO()
    PILImage.fromarray(image.cpu().numpy()).save(
        f, "PNG", quality=100)
    image = ImageWidget(value=f.getvalue())
    canvas.draw_image(image, 0, 0, canvas.width, canvas.height)

def _print_item_pixel_info(canvas, item, x, y):
    """helper function to print info of items produced by render"""
    if torch.is_tensor(item):
        assert len(item.shape) in [2, 3], f"item is of shape {item.shape}"
        item_height = item.shape[0]
        item_width = item.shape[1]
        if item_height == canvas.height and item_width == canvas.width:
            print(f"{item[y, x]}")
        else:
            scaled_x = int(x * item_width / canvas.width)
            scaled_y = int(y * item_height / canvas.height)
            print(f"{item[scaled_y, scaled_x]} (coords scaled to {scaled_x, scaled_y})")
    else:
        print(f"{item}")

class BaseIpyVisualizer(object):
    r"""Base class for ipython visualizer.

    To create a visualizer one must define the class attribute _WATCHED_EVENTS and
    the method :func:`_handle_event`.

    the method :func:`_handle_event` must use the methods
    :func:`self.render()` or :func:`self.fast_render()` to update the canvas

    You can overload the constructor
    (make sure to reuse the base class one so that ipycanvas and ipyevents are properly used)

    Args:
        height (int): height of the canvas.
        width (int): width of the canvas.
        camera (kal.render.camera.Camera): Camera used for the visualization.
        render (Callable):
            render function that take a :class:`kal.render.camera.Camera` as input.
            Must return a torch.ByteTensor as output,
            or a dictionary where the element 'img' is a torch.ByteTensor to be displayed,
            of shape :math:`(\text{output_height}, \text{output_width}, 3)`,
            height and width don't have to match canvas dimension.
        fast_render (optional, Callable):
            A faster rendering function that may be used when doing high frequency manipulation
            such as moving the camera with a mouse. Default: same than ``render``.
        watched_events (list of str):
            Events to be watched by the visualizer
            (see `ipyevents main documentation`_).
        max_fps (float):
            maximum framerate for handling consecutive events,
            this is useful when the rendering is slow to avoid freezes.
            Typically 24 fps is great when working on a local machine
            with :func:`render` close to real-time,
            and lower to 10 fps with slower rendering or network latency.

    .. _ipyevents main documentation: https://github.com/mwcraig/ipyevents/blob/main/docs/events.ipynb
    """

    def __init__(self, height, width, camera, render, fast_render=None,
                 watched_events=None, max_fps=None):
        self.canvas = Canvas(height=height, width=width)
        self.out = Output()
        assert len(camera) == 1, "only single camera supported for visualizer"
        self.camera = camera
        self.render = render
        if fast_render is None:
            self.fast_render = render
        else:
            self.fast_render = fast_render

        self._max_fps = max_fps
        self.current_output = None

        wait = 0 if max_fps is None else int(1000. / max_fps)

        self.event = Event(
            source=self.canvas,
            watched_events=watched_events,
            prevent_default_action=True,
            wait=wait,
        )
        self.event.on_dom_event(self._handle_event)


    def render_update(self):
        """Update the Canvas with :func:`render`"""
        with torch.no_grad():
            output = self.render(self.camera)
            if isinstance(output, dict):
                self.current_output = output
            elif isinstance(output, torch.Tensor):
                self.current_output = {'img': output}
            else:
                raise TypeError(f"render function output type ({type(output)}) unsupported")
            update_canvas(self.canvas, self.current_output['img'])

    def fast_render_update(self):
        """Update the Canvas with :func:`fast_render`"""
        with torch.no_grad():
            output = self.fast_render(self.camera)
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, dict):
                output = output['img']
            update_canvas(self.canvas, output)

    def show(self):
        """display the Canvas with interactive features"""
        self.render_update()
        display(self.canvas, self.out)

    def _print_pixel_all_infos(self, event):
        """print pixel all infos from event query"""
        self.out.clear_output()
        clamped_x = min(max(event["relativeX"], 0), self.canvas.width - 1)
        clamped_y = min(max(event["relativeY"], 0), self.canvas.height - 1)
        print(f'pixel coords: {clamped_x}, {clamped_y}')
        for key, item in self.current_output.items():
            print(key, end=': ')
            _print_item_pixel_info(self.canvas, item, clamped_x, clamped_y)


    @abstractmethod
    def _handle_event(self, event):
        pass

    @property
    def max_fps(self):
        """maximum fps for handling consecutive events"""
        return self._max_fps

    @max_fps.setter
    def max_fps(self, new_val):
        self._max_fps = new_val
        if new_val is None:
            self.event.wait = 0
        else:
            self.event.wait = int(1000. / self._max_fps)

@torch.jit.script
def make_quaternion_rotation(angle: float, vec: torch.Tensor):
    r"""Represent a rotation around axis as a quaternion.

    Args:
        angle (float): angle of rotation.
        vec (torch.Tensor):
            axis around which the rotation is done,
            of shape :math:`(\text{batch_size}, 3)`

    Returns:
        (torch.Tensor): A quaternion of shape :math:`(\text{batch_size}, 4)`
    """
    half_angle = angle / 2
    sin_half_angle = math.sin(half_angle)
    cos_half_angle = math.cos(half_angle)
    return torch.stack([
        vec[:, 0] * sin_half_angle,
        vec[:, 1] * sin_half_angle,
        vec[:, 2] * sin_half_angle,
        torch.full((vec.shape[0],), cos_half_angle, dtype=vec.dtype, device=vec.device)
    ], dim=-1)

@torch.jit.script
def conjugate(quat: torch.Tensor):
    r"""Return the conjugate of a quaternion.

    Args:
        quat (torch.Tensor): The quaternion, of shape :math:`(\text{batch_size}, 4)`.

    Returns:
        (torch.Tensor): the conjugate, of shape :math:`(\text{batch_size}, 4)`.
    """
    return torch.stack([-quat[:, 0], -quat[:, 1], -quat[:, 2], quat[:, 3]], dim=-1)

@torch.jit.script
def mulqv(q: torch.Tensor, v: torch.Tensor):
    r"""Return the product of a quaternion with a 3D vector.

    Support broadcasting.

    Args:
        q (torch.Tensor): The quaternion, of shape :math:`(\text{batch_size}, 4)`.
        v (torch.Tensor): The vector, of shape :math:`(\text{batch_size}, 3)`.

    Return:
        (torch.Tensor): A quaternion, of shape :math:`(\text{batch_size}, 4)`.
    """
    output = torch.stack([
        q[:, 3] * v[:, 0] + q[:, 1] * v[:, 2] - q[:, 2] * v[:, 1],
        q[:, 3] * v[:, 1] + q[:, 2] * v[:, 0] - q[:, 0] * v[:, 2],
        q[:, 3] * v[:, 2] + q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0],
        - q[:, 0] * v[:, 0] - q[:, 1] * v[:, 1] - q[:, 2] * v[:, 2],
    ], dim=-1)
    return output

@torch.jit.script
def mulqq(l: torch.Tensor, r: torch.Tensor):
    r"""Return the product of two quaternions.

    Support broadcasting.

    Args:
        l (torch.Tensor): The quaternion, of shape :math:`(\text{batch_size}, 4)`.
        r (torch.Tensor): The quaternion, of shape :math:`(\text{batch_size}, 4)`.

    Returns:
        (torch.Tensor): A quaternion, of shape :math:`(\text{batch_size}, 4)`.
    """
    output = torch.stack([
        l[:, 0] * r[:, 3] + l[:, 3] * r[:, 0] + l[:, 1] * r[:, 2] - l[:, 2] * r[:, 1],
        l[:, 1] * r[:, 3] + l[:, 3] * r[:, 1] + l[:, 2] * r[:, 0] - l[:, 0] * r[:, 2],
        l[:, 2] * r[:, 3] + l[:, 3] * r[:, 2] + l[:, 0] * r[:, 1] - l[:, 1] * r[:, 0],
        l[:, 3] * r[:, 3] - l[:, 0] * r[:, 0] - l[:, 1] * r[:, 1] - l[:, 2] * l[:, 2],
    ], dim=-1)
    return output

@torch.jit.script
def rotate_around_axis(point: torch.Tensor, angle: float, axis: torch.Tensor):
    r"""Compute the rotation of a point around an axis.

    Args:
        point (torch.Tensor): The point to be rotated, of shape :math:`(\text{batch_size}, 3)`.
        angle (float): The angle of rotation
        axis (torch.Tensor): The axis around which the point is revolving,
                             of shape :math:`(\text{batch_size}, 3)`.

    Returns:
        (torch.Tensor): The rotated point, of shape :math:`(\text{batch_size}, 3)`.
    """
    rot_q = make_quaternion_rotation(angle, axis)
    conj_q = conjugate(rot_q)
    w = mulqq(mulqv(rot_q, point), conj_q)
    return w[:, :-1]

class IpyTurntableVisualizer(BaseIpyVisualizer):
    r"""An interactive turntable visualizer that can display on jupyter notebook.

    You can move around with the mouse (using the left button), zoom with the wheel and
    get closer to the center with the wheel + control key.

    Args:
        height (int): height of the canvas.
        width (int): width of the canvas.
        camera (kal.render.camera.Camera):
            Camera used for the visualization.
            Note: The camera will be reoriented to look at ``focus_at``
            and with respect to ``world_up``.
        render (Callable):
            render function that take a :class:`kal.render.camera.Camera` as input.
            Must return a torch.ByteTensor as output,
            or a dictionary where the element 'img' is a torch.ByteTensor to be displayed,
            of shape :math:`(\text{output_height}, \text{output_width})`,
            height and width don't have to match canvas dimension.
        fast_render (optional, Callable):
            A faster rendering function that may be used when doing high frequency manipulation
            such as moving the camera with a mouse. Default: same than ``render``.
        focus_at (optional, torch.Tensor):
            The center of the turntable on which the camera is focusing on.
            Default: (0, 0, 0).
        world_up_axis (optional, int):
            The up axis of the world, in the coordinate system. Default: 1.
        zoom_sensitivity (float):
            Sensitivity of the wheel on zoom. Default: 1e-3.
        forward_sensitivity (float):
            Sensitivity of the wheel on forward. Default: 1e-3.
        mouse_sensitivity (float):
            Sensitivity of the mouse on movements. Default: 1.5.
        max_fps (optional, float):
            maximum framerate for handling consecutive events,
            this is useful when the rendering is slow to avoid freezes.
            Typically 24 fps is great when working on a local machine
            with :func:`render` close to real-time.
            And you lower to 10 fps with slower rendering or network latency.
            Default: 24 fps.
        update_only_on_release (bool):
            If true, the canvas won't be updated while the mouse button is pressed
            and only when it's released. To avoid freezes with very slow rendering functions.
            Default: False.
        additional_watched_events (optional, list of str):
            Additional events to be watched by the visualizer
            (see `ipyevents main documentation`_).
            To be used for customed events such as enabling / disabling a feature on a key press.
            ['wheel', 'mousedown', 'mouseup', 'mousemove', 'mouseleave'] are already watched.
            Default: None.
        additional_event_handler (optional, Callable):
            Additional event handler to be used for customed events such as
            enabling / disabling a feature on a key press.
            The Callable must take as input a tuple of (this visualizer object, the event).
            (see `ipyevents main documentation`_).

    Attributes:
        camera (kal.render.camera.Camera): The camera used for rendering.
        canvas (ipycanvas.Canvas): The canvas on which the rendering is copied to.
        out (ipywidgets.Output): An output where error and prints are displayed.
        render (Callable): The rendering function.
        fast_render (Callable)
        focus_at (torch.Tensor)
        world_up_axis (int)
        zoom_sensitivity (float)
        forward_sensitivity (float)
        mouse_sensitivity (float)
        max_fps (int)
        update_only_on_release (bool)

    Methods:
        show: display the Canvas with interactive features.
        render_update: Update the Canvas with :func:`render()`.
        fast_render_update: Update the Canvas with :func:`fast_render()`.

    .. _ipyevents main documentation: https://github.com/mwcraig/ipyevents/blob/main/docs/events.ipynb
    """
    def __init__(self,
                 height,
                 width,
                 camera,
                 render,
                 fast_render=None,
                 focus_at=None,
                 world_up_axis=1,
                 zoom_sensitivity=0.001,
                 forward_sensitivity=0.001,
                 mouse_sensitivity=1.5,
                 max_fps=24.,
                 update_only_on_release=False,
                 additional_watched_events=None,
                 additional_event_handler=None):

        with torch.no_grad():
            if focus_at is None:
                self.focus_at = torch.zeros((3,), device=camera.device)
            else:
                self.focus_at = focus_at

            up = torch.zeros((3,), device=camera.device)
            up[world_up_axis] = float(camera.cam_up().squeeze()[world_up_axis] >= 0) * 2. - 1.
            camera.extrinsics = CameraExtrinsics.from_lookat(
                eye=camera.cam_pos().squeeze(),
                at=self.focus_at,
                up=up,
                dtype=camera.dtype,
                device=camera.device,
            )

            self.position = None

            self.world_up_axis = world_up_axis
            self.zoom_sensitivity = zoom_sensitivity
            self.forward_sensitivity = forward_sensitivity
            self.mouse_scale = mouse_sensitivity * math.pi
            self.update_only_on_release = update_only_on_release

            watched_events = ['wheel', 'mousedown', 'mouseup', 'mousemove', 'mouseleave', 'mouseenter']
            if additional_watched_events is not None:
                watched_events += additional_watched_events
            self.additional_event_handler = additional_event_handler

            super().__init__(height, width, camera, render, fast_render,
                             watched_events, max_fps)

    def _move_turntable(self, amount_elevation, amount_azimuth):
        """Move the camera around a focus point as turntable

        Args:
            amount_elevation (float):
                Amount of elevation rotation, measured in radians.
            amount_azimuth (float):
                Amount of azimuth rotation, measure in radians.
        """
        radius = (self.camera.cam_pos().squeeze() - self.focus_at).norm(dim=-1, keepdim=True)
        # Rotates camera in normal direction to plane (world space)
        in_plane_amount = -amount_azimuth
        # Rotates camera in up-forward direction (camera space)
        pitch = -amount_elevation
        self.camera.t = torch.zeros(3, device=self.camera.device, dtype=self.camera.dtype)
        self.camera.rotate(pitch=pitch)
        translate = torch.eye(4, device=self.camera.device, dtype=self.camera.dtype)
        translate[:3, 3] = -self.focus_at
        rot_mat = torch.eye(4, device=self.camera.device, dtype=self.camera.dtype)
        if self.world_up_axis == 1:
            rot_mat[0, 0] = math.cos(in_plane_amount)
            rot_mat[0, 2] = -math.sin(in_plane_amount)
            rot_mat[2, 0] = math.sin(in_plane_amount)
            rot_mat[2, 2] = math.cos(in_plane_amount)
        elif self.world_up_axis == 2:
            rot_mat[0, 0] = math.cos(in_plane_amount)
            rot_mat[0, 1] = math.sin(in_plane_amount)
            rot_mat[1, 0] = -math.sin(in_plane_amount)
            rot_mat[1, 1] = math.cos(in_plane_amount)
        elif self.world_up_axis == 0:
            rot_mat[1, 1] = math.cos(in_plane_amount)
            rot_mat[1, 2] = math.sin(in_plane_amount)
            rot_mat[2, 1] = -math.sin(in_plane_amount)
            rot_mat[2, 2] = math.cos(in_plane_amount)
        view_matrix = self.camera.view_matrix().squeeze(0) @ rot_mat @ translate
        self.camera._backend.update(view_matrix)
        cam_forward = self.camera.cam_forward().squeeze()
        backward_dir = cam_forward / cam_forward.norm(dim=-1, keepdim=True)
        self.camera.translate(radius * backward_dir)

    def _safe_zoom(self, amount):
        r"""Applies a zoom on the camera by adjusting the lens.

        This function is different from :func:`kal.render.camera.CameraExtrinsics.zoom`
        in which the FOV is constrained by a sigmoid.

        Args:
            amount (float):
                Amount of adjustment.
                Mind the conventions -
                To zoom in, give a positive amount (decrease fov by amount -> increase focal length)
                To zoom out, give a negative amount (increase fov by amount -> decrease focal length)
        """
        fov_ratio = self.camera.fov_x / self.camera.fov_y
        fov_y_coeff = self.camera.fov_y / 180.
        inv_fov_y = torch.log(fov_y_coeff / (1 - fov_y_coeff))
        self.camera.fov_y = torch.sigmoid(inv_fov_y + amount) * 180.
        self.camera.fov_x = self.camera.fov_y * fov_ratio  # Make sure the view is not distorted

    def _safe_forward(self, amount):
        r"""Move the camera forward (or backward if negative)

        This functions is different from :func:`kal.render.camera.CameraExtrinsics.move_forward`
        in which the radius is restricted by :math:`new_radius = exp(log(old_radius) - amount)`

        Args:
            amount (float): Amout of adjustment (positive amount => move forward)
        """
        radius = (self.camera.cam_pos().squeeze() - self.focus_at).norm(dim=-1, keepdim=True)
        log_radius = torch.log(radius)
        new_radius = torch.exp(log_radius + amount)
        self.camera.move_forward(new_radius - radius)

    def _handle_event(self, event):
        with torch.no_grad():
            with self.out:
                process_event = True
                if self.additional_event_handler is not None:
                    process_event = self.additional_event_handler(self, event)
                if process_event:
                    if event['type'] == 'wheel':
                        if event['ctrlKey']:
                            self._safe_forward(event['deltaY'] * self.forward_sensitivity)
                        else:
                            self._safe_zoom(event['deltaY'] * self.zoom_sensitivity)
                        self.render_update()
                    elif event['type'] == 'mousedown':
                        self.position = (event['relativeX'], event['relativeY'])
                        # If the camera is upside down w.r.t to world we need to invert azimuth movement
                        self.sign = torch.sign(self.camera.cam_up()[0, self.world_up_axis, 0])
                    elif event['type'] in ['mouseup', 'mouseleave', 'mouseenter']:
                        self.render_update()
                        if event['type'] == 'mouseup' and event['button'] == 0:
                            self._print_pixel_all_infos(event)
                    elif event['type'] == 'mousemove' and event['buttons'] == 1:
                            dx = (self.mouse_scale *
                                  (event['relativeX'] - self.position[0]) / self.canvas.width)
                            dy = (self.mouse_scale *
                                  (event['relativeY'] - self.position[1]) / self.canvas.height)
                            self._move_turntable(dy, self.sign * dx)
                            self.position = (event['relativeX'], event['relativeY'])
                            if not self.update_only_on_release:
                                self.fast_render_update()

class IpyFirstPersonVisualizer(BaseIpyVisualizer):
    r"""An interactive first person visualizer that can display on jupyter notebook.

    You can move the orientation with the left button of the mouse,
    move the position of the camera with the right button of the mouse or the associated key,
    and zoom with the wheel.

    Args:
        height (int): height of the canvas.
        width (int): width of the canvas.
        camera (kal.render.camera.Camera):
            Camera used for the visualization.
        render (Callable):
            render function that take a :class:`kal.render.camera.Camera` as input.
            Must return a torch.ByteTensor as output,
            or a dictionary where the element 'img' is a torch.ByteTensor to be displayed,
            of shape :math:`(\text{output_height}, \text{output_width})`,
            height and width don't have to match canvas dimension.
        fast_render (optional, Callable):
            A faster rendering function that may be used when doing high frequency manipulation
            such as moving the camera with a mouse. Default: same than ``render``.
        world_up (optional, torch.Tensor):
            World up axis, of shape :math:`(3,)`. If provided the camera will be reoriented to avoid roll.
            Default: ``camera.cam_up()``.
        zoom_sensitivity (float):
            Sensitivity of the wheel on zoom. Default: 1e-3.
        rotation_sensitivity (float):
            Sensitivity of the mouse on rotations. Default: 0.4.
        translation_sensitivity (float):
            Sensitivity of the mouse on camera translation. Default: 1.
        key_move_sensitivity (float):
            Amount of camera movement on key press. Default 0.05.
        max_fps (optional, float):
            maximum framerate for handling consecutive events,
            this is useful when the rendering is slow to avoid freezes.
            Typically 24 fps is great when working on a local machine
            with :func:`render` close to real-time.
            And you lower to 10 fps with slower rendering or network latency.
            Default: 24 fps.
        up_key (str): key associated to moving up. Default 'i'.
        down_key (str): key associated to moving up. Default 'k'.
        left_key (str): key associated to moving up. Default 'j'.
        right_key (str): key associated to moving up. Default 'l'.
        forward_key (str): key associated to moving up. Default 'o'.
        backward_key (str): key associated to moving up. Default 'u'.
        update_only_on_release (bool):
            If true, the canvas won't be updated while the mouse button is pressed
            and only when it's released. To avoid freezes with very slow rendering functions.
            Default: False.
        additional_watched_events (optional, list of str):
            Additional events to be watched by the visualizer
            (see `ipyevents main documentation`_).
            To be used for customed events such as enabling / disabling a feature on a key press.
            ['wheel', 'mousedown', 'mouseup', 'mousemove', 'mouseleave'] are already watched.
            Default: None.
        additional_event_handler (optional, Callable):
            Additional event handler to be used for customed events such as
            enabling / disabling a feature on a key press.
            The Callable must take as input a tuple of (this visualizer object, the event).
            (see `ipyevents main documentation`_).

    Attributes:
        camera (kal.render.camera.Camera): The camera used for rendering.
        canvas (ipycanvas.Canvas): The canvas on which the rendering is copied to.
        out (ipywidgets.Output): An output where error and prints are displayed.
        render (Callable): The rendering function.
        fast_render (Callable)
        world_up (torch.Tensor)
        zoom_sensitivity (float)
        rotation_sensitivity (float)
        translation_sensitivity (float)
        key_move_sensitivity (float)
        max_fps (int)
        update_only_on_release (bool)

    Methods:
        show: display the Canvas with interactive features.
        render_update: Update the Canvas with :func:`render()`.
        fast_render_update: Update the Canvas with :func:`fast_render()`.

    .. _ipyevents main documentation: https://github.com/mwcraig/ipyevents/blob/main/docs/events.ipynb
    """

    def __init__(self,
                 height,
                 width,
                 camera,
                 render,
                 fast_render=None,
                 world_up=None,
                 zoom_sensitivity=0.001,
                 rotation_sensitivity=0.4,
                 translation_sensitivity=1.,
                 key_move_sensitivity=0.05,
                 max_fps=24.,
                 up_key='i',
                 down_key='k',
                 left_key='j',
                 right_key='l',
                 forward_key='o',
                 backward_key='u',
                 update_only_on_release=False,
                 additional_watched_events=None,
                 additional_event_handler=None):

        self.position = None

        with torch.no_grad():
            if world_up is None:
                self.world_up = torch.nn.functional.normalize(
                    camera.cam_up().clone().detach().squeeze(-1), dim=-1)
                self.world_right = torch.nn.functional.normalize(
                    camera.cam_right().clone().detach().squeeze(-1), dim=-1)
                self.elevation = torch.zeros((1,), device=camera.device, dtype=camera.dtype)
            else:
                self.world_up = torch.nn.functional.normalize(world_up, dim=-1)
                camera.extrinsics = CameraExtrinsics.from_lookat(
                    eye=camera.cam_pos().squeeze(),
                    at=(camera.cam_pos() - camera.cam_forward()).squeeze(),
                    up=self.world_up,
                    device=camera.device,
                    dtype=camera.dtype
                )
                if self.world_up.ndim == 1:
                    self.world_up = self.world_up.unsqueeze(0)

                self.world_right = camera.cam_right().squeeze(-1)
                self.elevation = torch.acos(torch.dot(
                    self.world_up.squeeze(), camera.cam_up().squeeze()
                )).reshape(1)
                if torch.dot(self.world_up.squeeze(), camera.cam_forward().squeeze()) >= 0:
                    self.elevation = -self.elevation
        self.azimuth = torch.zeros((1,), device=camera.device, dtype=camera.dtype)

        self.zoom_sensitivity = zoom_sensitivity
        self.rotation_scale = rotation_sensitivity * math.pi
        self.translation_sensitivity = translation_sensitivity
        self.key_move_sensitivity = key_move_sensitivity

        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key
        self.forward_key = forward_key
        self.backward_key = backward_key

        self.update_only_on_release = update_only_on_release

        watched_events = ['wheel', 'mousedown', 'mouseup', 'mousemove',
                          'mouseleave', 'mouseenter', 'contextmenu', 'keydown', 'keyup']
        if additional_watched_events is not None:
            watched_events += additional_watched_events
        self.additional_event_handler = additional_event_handler

        super().__init__(height, width, camera, render, fast_render,
                         watched_events, max_fps)

    def _safe_zoom(self, amount):
        r"""Applies a zoom on the camera by adjusting the lens.

        This function is different from :func:`kal.render.camera.CameraExtrinsics.zoom`
        in which the FOV is constrained by a sigmoid.

        Args:
            amount (float):
                Amount of adjustment.
                Mind the conventions -
                To zoom in, give a positive amount (decrease fov by amount -> increase focal length)
                To zoom out, give a negative amount (increase fov by amount -> decrease focal length)
        """
        fov_ratio = self.camera.fov_x / self.camera.fov_y
        fov_y_coeff = self.camera.fov_y / 180.
        inv_fov_y = torch.log(fov_y_coeff / (1 - fov_y_coeff))
        self.camera.fov_y = torch.sigmoid(inv_fov_y + amount) * 180.
        self.camera.fov_x = self.camera.fov_y * fov_ratio  # Make sure the view is not distorted

    def _first_person_rotate(self, move_azimuth, move_elevation):
        """Do a combination of rotations around camera-right axis and world up"""
        self.azimuth[:] = (self.azimuth + move_azimuth) % (2 * math.pi)
        self.elevation[:] = torch.clamp(self.elevation + move_elevation,
                                        -math.pi / 2., math.pi / 2.)
        cam_right = rotate_around_axis(self.world_right, self.azimuth, self.world_up)
        cam_up = rotate_around_axis(self.world_up, self.elevation, cam_right)
        cam_forward = torch.cross(cam_right, cam_up)
        world_rotation = torch.stack((cam_right, cam_up, cam_forward), dim=1)
        world_translation = -world_rotation @ self.camera.cam_pos()
        mat = self.camera.view_matrix()
        mat[:, :3, :3] = world_rotation
        mat[:, :3, 3] = world_translation.squeeze(-1)
        self.camera._backend.update(mat)

    def _handle_event(self, event):
        with torch.no_grad():
            with self.out:
                process_event = True
                if self.additional_event_handler is not None:
                    process_event = self.additional_event_handler(self, event)
                if process_event:
                    if event['type'] == 'wheel':
                        self._safe_zoom(event['deltaY'] * self.zoom_sensitivity)
                        self.render_update()
                    elif event['type'] == 'mousedown':
                        self.position = (event['relativeX'], event['relativeY'])
                    elif event['type'] in ['mouseup', 'mouseleave', 'mouseenter']:
                        self.render_update()
                        if event['type'] == 'mouseup' and event['button'] == 0:
                            self._print_pixel_all_infos(event)
                    elif event['type'] == 'mousemove':
                        if event['buttons'] == 1:
                            dx = (self.rotation_scale *
                                  (event['relativeX'] - self.position[0]) / self.canvas.width)
                            dy = (self.rotation_scale *
                                  (event['relativeY'] - self.position[1]) / self.canvas.height)
                            self._first_person_rotate(dx, dy)
                            self.position = (event['relativeX'], event['relativeY'])
                            if not self.update_only_on_release:
                                self.fast_render_update()
                        elif event['buttons'] == 2:
                            dx = (-self.translation_sensitivity *
                                  (event['relativeX'] - self.position[0]) / self.canvas.width)
                            dy = (self.translation_sensitivity *
                                  (event['relativeY'] - self.position[1]) / self.canvas.height)
                            self.camera.move_up(dy)
                            self.camera.move_right(dx)
                            self.position = (event['relativeX'], event['relativeY'])
                            if not self.update_only_on_release:
                                self.fast_render_update()
                    elif event['type'] == 'keydown':
                        if event['key'] == self.forward_key:
                            # Camera notion of forward is backward (OpenGL convention)
                            self.camera.move_forward(-self.key_move_sensitivity)
                            self.fast_render_update()
                        elif event['key'] == self.backward_key:
                            self.camera.move_forward(self.key_move_sensitivity)
                            self.fast_render_update()
                        elif event['key'] == self.up_key:
                            self.camera.move_up(self.key_move_sensitivity)
                            self.fast_render_update()
                        elif event['key'] == self.down_key:
                            self.camera.move_up(-self.key_move_sensitivity)
                            self.fast_render_update()
                        elif event['key'] == self.left_key:
                            self.camera.move_right(-self.key_move_sensitivity)
                            self.fast_render_update()
                        elif event['key'] == self.right_key:
                            self.camera.move_right(self.key_move_sensitivity)
                            self.fast_render_update()
                    elif event['type'] == 'keyup':
                        if event['key'] in [self.forward_key, self.backward_key, self.up_key,
                                            self.down_key, self.right_key, self.left_key]:
                            self.render_update()
