# ==============================================================================================================
# The following snippet demonstrates how to use the camera for generating a view-projection matrix
# as used in opengl shaders.
# ==============================================================================================================

import torch
import numpy as np
from kaolin.render.camera import Camera

# !!! This example is not runnable -- it is minimal to contain integration between the opengl shader and !!!
# !!! the camera matrix                                                                                  !!!
try:
    from glumpy import gloo
except:
    class DummyGloo(object):
        def Program(self, vertex, fragment):
            # see: https://glumpy.readthedocs.io/en/latest/api/gloo-shader.html#glumpy.gloo.Program
            return dict([])
    gloo = DummyGloo()


device = 'cuda'

camera = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    x0=0.0, y0=0.0,
    width=800, height=800,
    near=1e-2, far=1e2,
    dtype=torch.float64,
    device=device
)


vertex = """
            uniform mat4   u_viewprojection;
            attribute vec3 position;
            attribute vec4 color;
            varying vec4 v_color;
            void main()
            {
                v_color = color;
                gl_Position = u_viewprojection * vec4(position, 1.0f);
            } """

fragment = """
            varying vec4 v_color;
            void main()
            {
                gl_FragColor = v_color;
            } """

# Compile GL program
gl_program = gloo.Program(vertex, fragment)
gl_program["u_viewprojection"] = camera.view_projection_matrix()[0].cpu().numpy().T
