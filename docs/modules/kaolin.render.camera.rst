.. _kaolin.render.camera:

kaolin.render.camera
====================

Kaolin provides extensive camera API. For an overview, see the :ref:`Camera class docs <kaolin.render.camera.Camera>`.

API
---

Classes
^^^^^^^

* :ref:`Camera <kaolin.render.camera.Camera>`
* :ref:`CameraExtrinsics <kaolin.render.camera.CameraExtrinsics>`
* :ref:`CameraIntrinsics <kaolin.render.camera.CameraIntrinsics>`
* :ref:`PinholeIntrinsics <kaolin.render.camera.PinholeIntrinsics>`
* :ref:`OrthographicIntrinsics <kaolin.render.camera.OrthographicIntrinsics>`
* :ref:`ExtrinsicsRep <kaolin.render.camera.ExtrinsicsRep>`

.. _camera-conversions:

Camera Conversions
^^^^^^^^^^^^^^^^^^

Aligning camera conventions across different codebases can take time and care. Kaolin
ships with converters between :ref:`kaolin.render.camera.Camera <kaolin.render.camera.Camera>`
and camera conventions in several popular codebases, including:

* `nerfstudio gsplat <https://github.com/nerfstudio-project/gsplat>`_
* `INRIA gaussian splats <https://github.com/graphdeco-inria/gaussian-splatting>`_
* `polyscope <https://github.com/nmwsharp/polyscope>`_

Community contributions are welcome to expand this set.

.. autofunction:: kaolin.render.camera.kaolin_camera_to_gsplat_nerfstudio
.. autofunction:: kaolin.render.camera.gsplat_nerfstudio_camera_to_kaolin
.. autofunction:: kaolin.render.camera.kaolin_camera_to_gsplat_inria
.. autofunction:: kaolin.render.camera.gsplat_inria_camera_to_kaolin
.. autofunction:: kaolin.render.camera.kaolin_camera_to_polyscope
.. autofunction:: kaolin.render.camera.polyscope_camera_to_kaolin


Functions
^^^^^^^^^

.. automodule:: kaolin.render.camera
   :members:
   :exclude-members:
       Camera,
       CameraExtrinsics,
       CameraIntrinsics,
       PinholeIntrinsics,
       OrthographicIntrinsics,
       ExtrinsicsRep,
       kaolin_camera_to_gsplat_inria,
       gsplat_inria_camera_to_kaolin,
       kaolin_camera_to_gsplat_nerfstudio,
       gsplat_nerfstudio_camera_to_kaolin,
       kaolin_camera_to_polyscope,
       polyscope_camera_to_kaolin
   :undoc-members:
   :show-inheritance:

