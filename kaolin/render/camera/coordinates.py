# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

import torch

####################################################################################################################
# Convenience methods for converting from the default kaolin coordinate system to other common coordinate systems  #
# The default kaolin coordinate system is:                                                                         #
# right handed and cartesian (y axis pointing up, z pointing outwards of the screen):                              #
#                                                                                                                  #
#                                     Y                                                                            #
#                                     ^                                                                            #
#                                     |                                                                            #
#                                     |---------> X                                                                #
#                                    /                                                                             #
#                                  Z                                                                               #
####################################################################################################################


def blender_coords():
    """Blender world coordinates are right handed, with the z axis pointing upwards
    
    ::

        Z      Y
        ^    /
        |  /
        |---------> X

    """
    return torch.tensor([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]])


def opengl_coords():
    """Contemporary OpenGL doesn't enforce specific handedness on world coordinates.
    However it is common standard to define OpenGL world coordinates as right handed,
    with the y axis pointing upwards (cartesian)::

           Y
           ^
           |
           |---------> X
          /
        Z

    """
    return torch.tensor([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
