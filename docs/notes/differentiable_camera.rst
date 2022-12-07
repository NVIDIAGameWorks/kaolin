Differentiable Camera
*********************

.. _differentiable_camera:

Camera class
============

.. _camera_class:

:class:`kaolin.render.camera.Camera` is a one-stop class for all camera related differentiable / non-differentiable transformations.
Camera objects are represented by *batched* instances of 2 submodules:

    - :ref:`CameraExtrinsics <camera_extrinsics_class>`: The extrinsics properties of the camera (position, orientation).
      These are usually embedded in the view matrix, used to transform vertices from world space to camera space.
    - :ref:`CameraIntrinsics <camera_intrinsics_class>`: The intrinsics properties of the lens
      (such as field of view / focal length in the case of pinhole cameras).
      Intrinsics parameters vary between different lens type,
      and therefore multiple CameraIntrinsics subclasses exist,
      to support different types of cameras: pinhole / perspective, orthographic, fisheye, and so forth.
      For pinehole and orthographic lens, the intrinsics are embedded in a projection matrix.
      The intrinsics module can be used to transform vertices from camera space to Normalized Device Coordinates.

.. note::
    To avoid tedious invocation of camera functions through
    ``camera.extrinsics.someop()`` and ``camera.intrinsics.someop()``, kaolin overrides the ``__get_attributes__``
    function to forward any function calls of ``camera.someop()`` to
    the appropriate extrinsics / intrinsics submodule.

The entire pipeline of transformations can be summarized as (ignoring homogeneous coordinates)::

    World Space                                         Camera View Space
         V         ---CameraExtrinsics.transform()--->         V'          ---CameraIntrinsics.transform()---
    Shape~(B, 3)            (view matrix)                  Shape~(B, 3)                                     |
                                                                                                            |
                                                                           (linear lens: projection matrix) |
                                                                                  + homogeneus -> 3D        |
                                                                                                            V
                                                                                 Normalized Device Coordinates (NDC)
                                                                                            Shape~(B, 3)
    When using view / projection matrices, conversion to homogeneous coordinates is required.
    Alternatively, the `transform()` function takes care of such projections under the hood when needed.

How to apply transformations with kaolin's Camera:
    1. Linear camera types, such as the commonly used pinhole camera,
       support the :func:`view_projection_matrix()` method.
       The returned matrix can be used to transform vertices through pytorch's matrix multiplication, or even be
       passed to shaders as a uniform.
    2. All Cameras are guaranteed to support a general :func:`transform()` function
       which maps coordinates from world space to Normalized Device Coordinates space.
       For some lens types which perform non linear transformations,
       the :func:`view_projection_matrix()` is non-defined.
       Therefore the camera transformation must be applied through
       a dedicated function. For linear cameras,
       :func:`transform()` may use matrices under the hood.
    3. Camera parameters may also be queried directly.
       This is useful when implementing camera params aware code such as ray tracers.
How to control kaolin's Camera:
    - :class:`CameraExtrinsics`: is packed with useful methods for controlling the camera position and orientation:
      :func:`translate() <CameraExtrinsics.translate()>`,
      :func:`rotate() <CameraExtrinsics.rotate()>`,
      :func:`move_forward() <CameraExtrinsics.move_forward()>`,
      :func:`move_up() <CameraExtrinsics.move_up()>`,
      :func:`move_right() <CameraExtrinsics.move_right()>`,
      :func:`cam_pos() <CameraExtrinsics.cam_pos()>`,
      :func:`cam_up() <CameraExtrinsics.cam_up()>`,
      :func:`cam_forward() <CameraExtrinsics.cam_forward()>`,
      :func:`cam_up() <CameraExtrinsics.cam_up()>`.
    - :class:`CameraIntrinsics`: exposes a lens :func:`zoom() <CameraIntrinsics.zoom()>`
      operation. The exact functionality depends on the camera type.
How to optimize the Camera parameters:
    - Both :class:`CameraExtrinsics`: and :class:`CameraIntrinsics` maintain
      :class:`torch.Tensor` buffers of parameters which support pytorch differentiable operations.
    - Setting ``camera.requires_grad_(True)`` will turn on the optimization mode.
    - The :func:`gradient_mask` function can be used to mask out gradients of specific Camera parameters.

    .. note::
        :class:`CameraExtrinsics`: supports multiple representions of camera parameters
        (see: :func:`switch_backend <CameraExtrinsics.switch_backend()>`).
        Specific representations are better fit for optimization
        (e.g.: they maintain an orthogonal view matrix).
        Kaolin will automatically switch to using those representations when gradient flow is enabled
        For non-differentiable uses, the default representation may provide better
        speed and numerical accuracy.

Other useful camera properties:
    - Cameras follow pytorch in part, and support arbitrary ``dtype`` and ``device`` types through the
      :func:`to()`, :func:`cpu()`, :func:`cuda()`, :func:`half()`, :func:`float()`, :func:`double()`
      methods and :func:`dtype`, :func:`device` properties.
    - :class:`CameraExtrinsics`: and :class:`CameraIntrinsics`: individually support the :func:`requires_grad`
      property.
    - Cameras implement :func:`torch.allclose` for comparing camera parameters under controlled numerical accuracy.
      The operator ``==`` is reserved for comparison by ref.
    - Cameras support batching, either through construction, or through the :func:`cat()` method.

    .. note::
        Since kaolin's cameras are batched, the view/projection matrices are of shapes :math:`(\text{num_cameras}, 4, 4)`,
        and some operations, such as :func:`transform()` may return values as shapes of :math:`(\text{num_cameras}, \text{num_vectors}, 3)`.

Concluding remarks on coordinate systems and other confusing conventions:
    - kaolin's Cameras assume column major matrices, for example, the inverse view matrix (cam2world) is defined as:

      .. math::
          \begin{bmatrix}
              r1 & u1 & f1 & px \\
              r2 & u2 & f2 & py \\
              r3 & u3 & f3 & pz \\
              0 & 0 & 0 & 1
          \end{bmatrix}

      This sometimes causes confusion as the view matrix (world2cam) uses a transposed 3x3 submatrix component,
      which despite this transposition is still column major (observed through the last `t` column):

      .. math::
          \begin{bmatrix}
              r1 & r2 & r3 & tx \\
              u1 & u2 & u3 & ty \\
              f1 & f2 & f3 & tz \\
              0 & 0 & 0 & 1
          \end{bmatrix}

    - kaolin's cameras do not assume any specific coordinate system for the camera axes. By default, the
      right handed cartesian coordinate system is used. Other coordinate systems are supported through
      :func:`change_coordinate_system() <CameraExtrinsics.change_coordinate_system()>`
      and the ``coordinates.py`` module::

            Y
            ^
            |
            |---------> X
           /
         Z        - kaolin's NDC space is assumed to be left handed (depth goes inwards to the screen).

      The default range of values is [-1, 1].

CameraExtrinsics class
======================

.. _camera_extrinsics_class:

    :class:`kaolin.render.camera.CameraExtrinsics` holds the extrinsics parameters of a camera: position and orientation in space.

    This class maintains the view matrix of camera, used to transform points from world coordinates
    to camera / eye / view space coordinates.

    This view matrix maintained by this class is column-major, and can be described by the 4x4 block matrix:

    .. math::

        \begin{bmatrix}
            R & t \\
            0 & 1
        \end{bmatrix}

    where **R** is a 3x3 rotation matrix and **t** is a 3x1 translation vector for the orientation and position
    respectively.

    This class is batched and may hold information from multiple cameras.

    :class:`CameraExtrinsics` relies on a dynamic representation backend to manage the tradeoff between various choices
    such as speed, or support for differentiable rigid transformations.
    Parameters are stored as a single tensor of shape :math:`(\text{num_cameras}, K)`,
    where K is a representation specific number of parameters.
    Transformations and matrices returned by this class support differentiable torch operations,
    which in turn may update the extrinsic parameters of the camera::

                                 convert_to_mat
            Backend                 ---- >            Extrinsics
        Representation R                             View Matrix M
        Shape (num_cameras, K),                    Shape (num_cameras, 4, 4)
                                    < ----
                                 convert_from_mat

    .. note::

        Unless specified manually with :func:`switch_backend`,
        kaolin will choose the optimal representation backend depending on the status of ``requires_grad``.
    .. note::

        Users should be aware, but not concerned about the conversion from internal representations to view matrices.
        kaolin performs these conversions where and if needed.

    Supported backends:

        - **"matrix_se3"**\: A flattened view matrix representation, containing the full information of
          special euclidean transformations (translations and rotations).
          This representation is quickly converted to a view matrix, but differentiable ops may cause
          the view matrix to learn an incorrect, non-orthogonal transformation.
        - **"matrix_6dof_rotation"**\: A compact representation with 6 degrees of freedom, ensuring the view matrix
          remains orthogonal under optimizations. The conversion to matrix requires a single Gram-Schmidt step.

        .. seealso::

            `On the Continuity of Rotation Representations in Neural Networks, Zhou et al. 2019
            <https://arxiv.org/abs/1812.07035>`_

    Unless stated explicitly, the definition of the camera coordinate system used by this class is up to the
    choice of the user.
    Practitioners should be mindful of conventions when pairing the view matrix managed by this class with a projection
    matrix.

CameraIntrinsics class
======================

.. _camera_intrinsics_class:

    :class:`kaolin.render.camera.CameraIntrinsics` holds the intrinsics parameters of a camera:
    how it should project from camera space to normalized screen / clip space.

    The instrinsics are determined by the camera type, meaning parameters may differ according to the lens structure.
    Typical computer graphics systems commonly assume the intrinsics of a pinhole camera (see: :class:`PinholeIntrinsics` class).
    One implication is that some camera types do not use a linear projection (i.e: Fisheye lens).

    There are therefore numerous ways to use CameraIntrinsics subclasses:

        1. Access intrinsics parameters directly.
        This may typically benefit use cases such as ray generators.
        2. The :func:`transform()` method is supported by all CameraIntrinsics subclasses,
        both linear and non-linear transformations, to project vectors from camera space to normalized screen space.
        This method is implemented using differential pytorch operations.
        3. Certain CameraIntrinsics subclasses which perform linear projections, may expose the transformation matrix
        via dedicated methods.
        For example, :class:`PinholeIntrinsics` exposes a :func:`projection_matrix()` method.
        This may typically be useful for rasterization based rendering pipelines (i.e: OpenGL vertex shaders).

    This class is batched and may hold information from multiple cameras.
    Parameters are stored as a single tensor of shape :math:`(\text{num_cameras}, K)` where K is the number of
    intrinsic parameters.

    currently there are two subclasses of intrinsics: :class:`kaolin.render.camera.OrthographicIntrinsics` and
    :class:`kaolin.render.camera.PinholeIntrinsics`.

API Documentation:
------------------

* Check all the camera classes and functions at the :ref:`API documentation<kaolin.render.camera>`.

