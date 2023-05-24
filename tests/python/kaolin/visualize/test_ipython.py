# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import copy
import math
import random

import torch
import numpy as np
import pytest

import kaolin

class DummyRenderer():
    def __init__(self, height, width, value, output_dict=False):
        self.height = height
        self.width = width
        self.value = value
        self.render_count = 0
        self.event_count = 0
        self.output_dict = output_dict

    def __call__(self, camera):
        self.render_count += 1
        img = torch.full((self.height, self.width, 3), self.value,
                         device=camera.device, dtype=torch.uint8) 
        if self.output_dict:
            return {
                'img': img,
                'a': 1
            }
        else:
            return img

@pytest.mark.parametrize('height,width', [(16, 16), (32, 32)])
@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('output_dict', [False, True])
class TestVisualizers:

    @pytest.fixture(autouse=True)
    def camera(self, height, width, device):
        return kaolin.render.camera.Camera.from_args(
            eye=(torch.rand((3,)) - 0.5) * 10,
            at=(torch.rand((3,)) - 0.5) * 10,
            up=(torch.rand((3,)) - 0.5) * 10,
            fov=random.uniform(0.1, math.pi - 0.1),
            height=height,
            width=width,
            dtype=torch.float,
            device=device
        )

    @pytest.fixture(autouse=True)
    def renderer(self, height, width, output_dict):
        return DummyRenderer(
            height, width, 0, output_dict
        )

    @pytest.fixture(autouse=True)
    def fast_renderer(self, height, width, output_dict):
        return DummyRenderer(
            int(height / 4), int(width / 4), 255, output_dict
        )

    #TODO(cfujitsang): can't find a way to test max_fps
    @pytest.mark.parametrize('with_fast_renderer', [True, False])
    @pytest.mark.parametrize('world_up_axis', [0, 1])
    @pytest.mark.parametrize('with_focus_at', [True, False])
    @pytest.mark.parametrize('with_sensitivity', [True, False])
    @pytest.mark.parametrize('with_additional_event', [True, False])
    @pytest.mark.parametrize('update_only_on_release', [True, False])
    def test_turntable_visualizer(
            self, height, width, device, camera, renderer, fast_renderer, world_up_axis,
            with_focus_at, with_sensitivity, with_additional_event,
            update_only_on_release, with_fast_renderer):
        kwargs = {}

        if with_focus_at:
            focus_at = torch.rand((3,), device=camera.device, dtype=camera.dtype) - 0.5 * 10
            kwargs['focus_at'] = focus_at
        else:
            focus_at = torch.zeros((3,), device=camera.device, dtype=camera.dtype)

        if with_sensitivity:
            zoom_sensitivity = 0.01
            forward_sensitivity = 0.01
            mouse_sensitivity = 2.
            kwargs['zoom_sensitivity'] = zoom_sensitivity
            kwargs['forward_sensitivity'] = forward_sensitivity
            kwargs['mouse_sensitivity'] = mouse_sensitivity
        else:
            zoom_sensitivity = 0.001
            forward_sensitivity = 0.001
            mouse_sensitivity = 1.5

        global event_count
        event_count = 0
        if with_additional_event:
            def additional_event_handler(visualizer, event):
                with visualizer.out:
                    if event['type'] == 'mousedown' and event['buttons'] == 3:
                        global event_count
                        event_count += 1
                        return False
                return True
            kwargs['additional_event_handler'] = additional_event_handler
            kwargs['additional_watched_events'] = []

        if with_fast_renderer:
            kwargs['fast_render'] = fast_renderer

        viz = kaolin.visualize.IpyTurntableVisualizer(
            height,
            width,
            copy.deepcopy(camera),
            renderer,
            world_up_axis=world_up_axis,
            update_only_on_release=update_only_on_release,
            **kwargs
        )
        expected_render_count = 0
        expected_fast_render_count = 0
        def check_counts():
            if with_fast_renderer:
                assert renderer.render_count == expected_render_count
                assert fast_renderer.render_count == expected_fast_render_count
            else:
                assert renderer.render_count == expected_render_count + expected_fast_render_count
        assert torch.allclose(viz.focus_at, focus_at)
        check_counts()
        assert viz.canvas.height == height
        assert viz.canvas.width == width

        # Test reorientation at ctor
        assert torch.allclose(viz.camera.cam_pos(), camera.cam_pos(), atol=1e-5, rtol=1e-5), \
            "After ctor: camera moved"
        signed_world_up = torch.zeros((3,), device=camera.device)
        signed_world_distance = float(camera.cam_up().squeeze()[world_up_axis] >= 0) * 2. - 1.
        signed_world_up[world_up_axis] = signed_world_distance
        assert torch.dot(signed_world_up, viz.camera.cam_up().squeeze()) >= 0, \
            "After ctor: camera up is wrong direction"
        assert torch.dot(signed_world_up, viz.camera.cam_right().squeeze()) == 0, \
            "After ctor: camera right is not perpendicular to the world up"

        expected_cam_forward = torch.nn.functional.normalize(viz.focus_at - camera.cam_pos().squeeze(), dim=-1)
        assert torch.allclose(
            torch.dot(-viz.camera.cam_forward().squeeze(), expected_cam_forward),
            torch.ones((1,), device=camera.device)
        ), "After ctor: camera is not looking at focus_at"

        ctor_camera = copy.deepcopy(viz.camera)
        ref_radius = torch.linalg.norm(
            viz.focus_at - ctor_camera.cam_pos().squeeze(),
            dim=-1
        )
        signed_world_right = torch.zeros((3,), device=camera.device)
        signed_world_right[world_up_axis - 1] = signed_world_distance
        signed_world_forward = torch.zeros((3,), device=camera.device)
        signed_world_forward[world_up_axis - 2] = signed_world_distance
        ctor_cam_2d_pos = torch.stack([
            viz.camera.cam_pos().squeeze()[world_up_axis - 1],
            viz.camera.cam_pos().squeeze()[world_up_axis - 2],
        ], dim=0)

        try:
            viz.show()
        except NameError: # show() use "display()" that is builtin only in ipython
            pass

        expected_render_count += 1
        check_counts()
        assert torch.equal(ctor_camera.view_matrix(), viz.camera.view_matrix()), \
            "After .show(): camera have moved"
        assert torch.equal(ctor_camera.params, viz.camera.params), \
            "After .show(): camera intrinsics have changed"


        from_x = random.randint(0, width)
        from_y = random.randint(0, height)
        viz._handle_event({'type': 'mousedown', 'relativeX': from_x, 'relativeY': from_y, 'buttons': 1})
        check_counts()
        assert torch.equal(ctor_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mousedown: camera have moved"
        assert torch.equal(ctor_camera.params, viz.camera.params), \
            "After mousedown: camera intrinsics have changed"

        to_x = random.randint(0, width)
        while to_x != from_x:
            to_x = random.randint(0, width)

        to_y = random.randint(0, height)
        while to_y != from_y:
            to_y = random.randint(0, height)

        viz._handle_event({'type': 'mousemove', 'relativeX': to_x, 'relativeY': to_y, 'buttons': 1})
        if not update_only_on_release:
            expected_fast_render_count += 1
        check_counts()
        cur_radius = torch.linalg.norm(
            viz.focus_at - viz.camera.cam_pos().squeeze(),
            dim=-1
        )
        assert torch.allclose(cur_radius, ref_radius)
        cur_focus_at = (
            viz.camera.cam_pos() - viz.camera.cam_forward() * cur_radius
        ).squeeze()
        assert torch.allclose(viz.focus_at, cur_focus_at, atol=1e-5, rtol=1e-5)

        azimuth_diff = mouse_sensitivity * (to_x - from_x) * math.pi / viz.canvas.width
        elevation_diff = mouse_sensitivity * (to_y - from_y) * math.pi / viz.canvas.height

        cur_cam_pos = kaolin.visualize.ipython.rotate_around_axis(
            ctor_camera.cam_pos().squeeze(-1) - focus_at.unsqueeze(0),
            -azimuth_diff,
            signed_world_up.unsqueeze(0)
        )
        cur_cam_pos = kaolin.visualize.ipython.rotate_around_axis(
            cur_cam_pos,
            -elevation_diff,
            viz.camera.cam_right().squeeze(-1),
        ) + focus_at.unsqueeze(0)
        assert torch.allclose(cur_cam_pos, viz.camera.cam_pos().squeeze(-1),
                              atol=1e-4, rtol=1e-4)
        cur_camera = copy.deepcopy(viz.camera)
        viz._handle_event({'type': 'mouseup', 'button': 0, 'buttons': 1,
                           'relativeX': to_x, 'relativeY': to_y})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mouseup: camera have moved"
        assert torch.equal(cur_camera.params, viz.camera.params), \
            "After mouseup: camera intrinsics have changed"
        wheel_amount = 120 * random.randint(1, 10)
        viz._handle_event({'type': 'wheel', 'deltaY': wheel_amount, 'ctrlKey': False})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After unzoom: camera have moved"
        assert viz.camera.fov_x > cur_camera.fov_x, \
            "After unzoom: Didn't unzoom"
        assert viz.camera.fov_x < 180.
        cur_camera = copy.deepcopy(viz.camera)
        viz._handle_event({'type': 'wheel', 'deltaY': -2. * wheel_amount, 'ctrlKey': False})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After zoom: camera have moved"
        assert viz.camera.fov_x < cur_camera.fov_x, \
            "After zoom: Didn't zoom"
        assert viz.camera.fov_x > 0.
        cur_camera = copy.deepcopy(viz.camera)
        viz._handle_event({'type': 'wheel', 'deltaY': -wheel_amount, 'ctrlKey': True})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.params, viz.camera.params), \
            "After move forward: camera intrinsics have changed"
        normalized_distance = torch.nn.functional.normalize(
            cur_camera.cam_pos().squeeze() - viz.camera.cam_pos().squeeze(),
            dim=-1
        )
        assert torch.allclose(cur_camera.cam_forward(), viz.camera.cam_forward()), \
            "After move forward: camera have changed cam_forward()"
        assert torch.allclose(cur_camera.cam_up(), viz.camera.cam_up()), \
            "After move forward: camera have change cam_up()"
        assert torch.allclose(normalized_distance, cur_camera.cam_forward().squeeze(),
                              atol=1e-5, rtol=1e-5), \
             "After move forward: camera haven't moved forward"
        assert torch.all(torch.sign(focus_at - cur_camera.cam_pos().squeeze()) *
                         torch.sign(focus_at - viz.camera.cam_pos().squeeze()) >= 0.), \
            "After move forward: camera have crossed the focusing point"

        assert event_count == 0
        viz._handle_event({'type': 'mousedown', 'buttons': 3, 'relativeX': 0, 'relativeY': 0})
        check_counts()
        if with_additional_event:
            assert event_count == 1
        else:
            assert event_count == 0

    @pytest.mark.parametrize('with_fast_renderer', [True, False])
    @pytest.mark.parametrize('with_world_up', [True, False])
    @pytest.mark.parametrize('with_sensitivity', [True, False])
    @pytest.mark.parametrize('with_additional_event', [True, False])
    @pytest.mark.parametrize('update_only_on_release', [True, False])
    def test_first_person_visualizer(
            self, height, width, device, camera, renderer, fast_renderer,
            with_fast_renderer, with_world_up, with_sensitivity,
            with_additional_event, update_only_on_release):
        kwargs = {}
        if with_fast_renderer:
            kwargs['fast_render'] = fast_renderer
        if with_world_up:
            world_up = torch.nn.functional.normalize(
                torch.rand((3,), device=camera.device, dtype=camera.dtype),
                dim=-1
            )
            kwargs['world_up'] = world_up
        else:
            world_up = camera.cam_up().squeeze()

        if with_sensitivity:
            rotation_sensitivity = 0.1
            translation_sensitivity = 0.1
            key_move_sensitivity = 0.1
            zoom_sensitivity = 0.01
            kwargs['rotation_sensitivity'] = rotation_sensitivity
            kwargs['translation_sensitivity'] = translation_sensitivity
            kwargs['key_move_sensitivity'] = key_move_sensitivity
            kwargs['zoom_sensitivity'] = zoom_sensitivity

            up_key = 'w'
            down_key = 's'
            left_key = 'a'
            right_key = 'd'
            forward_key = 'e'
            backward_key = 'q'
            kwargs['up_key'] = up_key
            kwargs['down_key'] = down_key
            kwargs['left_key'] = left_key
            kwargs['right_key'] = right_key
            kwargs['forward_key'] = forward_key
            kwargs['backward_key'] = backward_key
        else:
            rotation_sensitivity = 0.4
            translation_sensitivity = 1.
            key_move_sensitivity = 0.05
            zoom_sensitivity= 0.001
            up_key = 'i'
            down_key = 'k'
            left_key = 'j'
            right_key = 'l'
            forward_key = 'o'
            backward_key = 'u'

        global event_count
        event_count = 0
        if with_additional_event:
            def additional_event_handler(visualizer, event):
                with visualizer.out:
                    if event['type'] == 'mousedown' and event['buttons'] == 3:
                        global event_count
                        event_count += 1
                        return False
                return True
            kwargs['additional_event_handler'] = additional_event_handler
            kwargs['additional_watched_events'] = ['mouseenter']

        viz = kaolin.visualize.IpyFirstPersonVisualizer(
            height,
            width,
            copy.deepcopy(camera),
            renderer,
            update_only_on_release=update_only_on_release,
            **kwargs
        )
        expected_render_count = 0
        expected_fast_render_count = 0
        def check_counts():
            if with_fast_renderer:
                assert renderer.render_count == expected_render_count
                assert fast_renderer.render_count == expected_fast_render_count
            else:
                assert renderer.render_count == expected_render_count + expected_fast_render_count
        check_counts()
        assert viz.canvas.height == height
        assert viz.canvas.width == width

        # Test reorientation at ctor
        expected_extrinsics = kaolin.render.camera.CameraExtrinsics.from_lookat(
            eye=camera.cam_pos().squeeze(),
            at=(camera.cam_pos().squeeze() - camera.cam_forward().squeeze()),
            up=world_up,
            device=camera.device,
            dtype=camera.dtype
        )
        assert torch.allclose(expected_extrinsics.view_matrix(), viz.camera.view_matrix(),
                              atol=1e-5, rtol=1e-5)
        ctor_camera = copy.deepcopy(viz.camera)
        
        try:
            viz.show()
        except NameError: # show() use "display()" that is builtin only in ipython
            pass

        expected_render_count += 1
        check_counts()
        assert torch.equal(ctor_camera.view_matrix(), viz.camera.view_matrix()), \
            "After .show(): camera have moved"
        assert torch.equal(ctor_camera.params, viz.camera.params), \
            "After .show(): camera intrinsics have changed"

        from_x = random.randint(0, width)
        from_y = random.randint(0, height)
        viz._handle_event({'type': 'mousedown', 'relativeX': from_x, 'relativeY': from_y, 'buttons': 1})
        check_counts()
        assert torch.equal(ctor_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mousedown: camera have moved"
        assert torch.equal(ctor_camera.params, viz.camera.params), \
            "After mousedown: camera intrinsics have changed"

        to_x = random.randint(0, width)
        while to_x != from_x:
            to_x = random.randint(0, width)

        to_y = random.randint(0, height)
        while to_y != from_y:
            to_y = random.randint(0, height)

        ctor_elevation = viz.elevation

        viz._handle_event({'type': 'mousemove', 'relativeX': to_x, 'relativeY': to_y, 'buttons': 1})
        if not update_only_on_release:
            expected_fast_render_count += 1
        check_counts()

        azimuth_diff = rotation_sensitivity * (to_x - from_x) * math.pi / viz.canvas.width
        elevation_diff = rotation_sensitivity * (to_y - from_y) * math.pi / viz.canvas.height
        _elevation = ctor_elevation + elevation_diff
        if _elevation > math.pi / 2.:
            elevation_diff = math.pi / 2. - ctor_elevation
        if _elevation < -math.pi / 2.:
            elevation_diff = -math.pi / 2. - ctor_elevation
        assert viz.elevation == ctor_elevation + elevation_diff

        cur_cam_forward = kaolin.visualize.ipython.rotate_around_axis(
            ctor_camera.cam_forward().squeeze(-1),
            -azimuth_diff,
            world_up.unsqueeze(0)
        )
        cur_cam_right = kaolin.visualize.ipython.rotate_around_axis(
            ctor_camera.cam_right().squeeze(-1),
            -azimuth_diff,
            world_up.unsqueeze(0)
        )
        cur_cam_up = kaolin.visualize.ipython.rotate_around_axis(
            ctor_camera.cam_up().squeeze(-1),
            -azimuth_diff,
            world_up.unsqueeze(0)
        )

        cur_cam_forward = kaolin.visualize.ipython.rotate_around_axis(
            cur_cam_forward,
            -elevation_diff,
            cur_cam_right,
        )
        cur_cam_up = kaolin.visualize.ipython.rotate_around_axis(
            cur_cam_up,
            -elevation_diff,
            cur_cam_right,
        )

        assert torch.allclose(ctor_camera.cam_pos().squeeze(-1), viz.camera.cam_pos().squeeze(-1),
                              atol=1e-4, rtol=1e-4)
        assert torch.allclose(cur_cam_right, viz.camera.cam_right().squeeze(-1),
                              atol=1e-4, rtol=1e-4)
        assert torch.allclose(cur_cam_forward, viz.camera.cam_forward().squeeze(-1),
                              atol=1e-4, rtol=1e-4)
        assert torch.allclose(cur_cam_up, viz.camera.cam_up().squeeze(-1),
                              atol=1e-4, rtol=1e-4)
        cur_camera = copy.deepcopy(viz.camera)

        viz._handle_event({'type': 'mouseup', 'button': 0,
                           'relativeX': to_x, 'relativeY': to_y})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mouseup: camera have moved"
        assert torch.equal(cur_camera.params, viz.camera.params), \
            "After mouseup: camera intrinsics have changed"

        from_x = random.randint(0, width)
        from_y = random.randint(0, height)

        viz._handle_event({
            'type': 'mousedown', 'relativeX': from_x, 'relativeY': from_y, 'buttons': 2
        })
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mousedown: camera have moved"
        assert torch.equal(cur_camera.params, viz.camera.params), \
            "After mousedown: camera intrinsics have changed"

        to_x = random.randint(0, width)
        while to_x != from_x:
            to_x = random.randint(0, width)

        to_y = random.randint(0, height)
        while to_y != from_y:
            to_y = random.randint(0, height)

        viz._handle_event({
            'type': 'mousemove', 'relativeX': to_x, 'relativeY': to_y, 'buttons': 2
        })
        if not update_only_on_release:
            expected_fast_render_count += 1
        check_counts()

        cur_camera.move_up(translation_sensitivity * (to_y - from_y) / height)
        cur_camera.move_right(-translation_sensitivity * (to_x - from_x) / width)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'mouseup', 'button': 1,
                           'relativeX': to_x, 'relativeY': to_y})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After mouseup: camera have moved"
        assert torch.equal(cur_camera.params, viz.camera.params), \
            "After mouseup: camera intrinsics have changed"

        viz._handle_event({'type': 'keydown', 'key': up_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_up(key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': up_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': down_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_up(-key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': down_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': left_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_right(-key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': left_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': right_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_right(key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': right_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': forward_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_forward(-key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': forward_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': backward_key})
        expected_fast_render_count += 1
        check_counts()
        cur_camera.move_forward(key_move_sensitivity)
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': backward_key})
        expected_render_count += 1
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keydown', 'key': 'x'})
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        viz._handle_event({'type': 'keyup', 'key': 'x'})
        check_counts()
        assert torch.allclose(cur_camera.view_matrix(), viz.camera.view_matrix())
        assert torch.allclose(cur_camera.params, viz.camera.params)

        wheel_amount = 120 * random.randint(1, 10)
        viz._handle_event({'type': 'wheel', 'deltaY': wheel_amount})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After unzoom: camera have moved"
        assert viz.camera.fov_x > cur_camera.fov_x, \
            "After unzoom: Didn't unzoom"
        assert viz.camera.fov_x < 180.
        cur_camera = copy.deepcopy(viz.camera)
        viz._handle_event({'type': 'wheel', 'deltaY': -2. * wheel_amount})
        expected_render_count += 1
        check_counts()
        assert torch.equal(cur_camera.view_matrix(), viz.camera.view_matrix()), \
            "After zoom: camera have moved"
        assert viz.camera.fov_x < cur_camera.fov_x, \
            "After zoom: Didn't zoom"
        assert viz.camera.fov_x > 0.

        assert event_count == 0
        viz._handle_event({'type': 'mousedown', 'buttons': 3, 'relativeX': 0, 'relativeY': 0})
        check_counts()
        if with_additional_event:
            assert event_count == 1
        else:
            assert event_count == 0

