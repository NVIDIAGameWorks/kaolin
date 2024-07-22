Easy PBR Shader (USD, gltf, obj)
********************************

.. _pbr_shader:

.. image:: ../img/easy_render_urchin.jpg

.. raw:: html

   <div style="font-size: 75%; margin-top: -10px; margin-bottom: 20px;">
   Differentiable rendering of an in-the-wild  <a src="https://sketchfab.com/models/c5c2ad0175f943969abc4d2368c0d2ff/embed">Sea Urchin Shell</a> by <a href="https://sketchfab.com/drakery">Drakery</a> from <a href="https://www.sketchfab.com/">Sketchfab</a>,
   with no specialized geometry or material preprocessing.
   </div>

It can be challenging to compose custom differentiable rendering pipelines, especially given the wide
range of conventions, and this step can take some ramp-up time for research projects.
With Kaolin Library v0.16.0, we are piloting a standard differentiable
shader :any:`kaolin.render.easy_render.render_mesh` that works out-of-the-box for many meshes,
supporting :ref:`Spherical Gaussian lighting <differentiable_lighting>`, and a partial set of physics based rendering (PBR) material maps, including albedo
textures, roughness and specular workflows and normal maps.

See End-to-end Tutorial: `examples/tutorial/easy_mesh_render.ipynb <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/easy_mesh_render.ipynb>`_.

.. Caution:: There are many differentiable rendering variants, and this particular module has not yet been research-tested for specific applications. We did find it robust across many in-the-wild meshes any welcome your feedback on our `GitHub <https://github.com/NVIDIAGameWorks/kaolin/issues>`_. **Please cite Kaolin if this is useful in your research** (`citation <https://github.com/NVIDIAGameWorks/kaolin?tab=readme-ov-file#citation>`_).


Consistent Convensions for USD, GLTF, GLB, OBJ
==============================================

One of the issues with differentiably rendering existing meshes is that input file formats as well as readers into tensors used in deep learning frameworks like PyTorch have a wide range of
convensions. Since v0.16.0, Kaolin has started to make the effort to make meshes imported from ``.USD``, ``.GLTF``, ``.obj`` using
:any:`kaolin.io.mesh.import_mesh` follow consistent
convensions, both for imported geometry, represented as :class:`kaolin.rep.SurfaceMesh` and materials, represented as :class:`kaolin.render.materials.PBRMaterial`.
Given staggering number of subtleties, we cannot guarantee consistency or full support for many aspects of these formats, but
latest release is a step toward enabling 3D DL researchers to more easily integrate real meshes into
their research, rather than creating them from scratch. For example, we would like to enable research into iterative workflows for artists, allowing use of
traditional authoring tools in conjunction with new AI algorithms. To load a mesh into PyTorch (see also :ref:`Working With Surface Meshes <surface_meshes>`)::

   mesh = kaolin.io.import_mesh('usd_gltf_or_obj_path', triangulate=True).cuda()


Standard Differentiable PBR Shader
==================================

To render imported (or constructed) triangular meshes (note that you can specify ``triangluate=True`` on import), simply call :any:`kaolin.render.easy_render.render_mesh`
to output full composited pass, as well as normals, albedo, specular, alpha and face_id passes. You will first need to create :class:`kaolin.render.camera.Camera`
(see :ref:`Differentiable Camera <differentiable_camera>`) or use a default::

   # Use default camera, lighting or construct your own
   camera = kaolin.render.easy_render.default_camera(512).cuda()
   lighting = kaolinl.render.easy_render.default_lighting().cuda()

   # Render
   render_res = kal.render.easy_render.render_mesh(in_cam, mesh, lighting=lighting, **kwargs)

.. Tip::
    **Both** DIB-R and nvdiffrast :ref:`pipelines <diff_render>` are supported, by passing in ``backend="cuda"`` and ``backend="nvdiffrast"``, respectively.
    See :any:`documentation <kaolin.render.easy_render.render_mesh>` for all available options.


A Note on the Backends
======================

See :ref:`Differentiable Rendering <diff_render>` for background on different backends. As noted above, the :any:`easy_render.render_mesh <kaolin.render.easy_render.render_mesh>`
supports two back ends. We recommend using `nvdiffrast <https://github.com/NVlabs/nvdiffrast>`_. If this library is installed, a default OpenGL based context will be created
and reused on the provided GPU device. It is also possible to pass in the desirable context, or to configure the behavior of devault context constructions through
methods like :any:`nvdiffrast_use_cuda() <kaolin.render.mesh.nvdiffrast_context.nvdiffrast_use_cuda>`. See :py:mod:`kaolin.render.mesh.nvdiffrast_context`.
Note that not all nvdiffrast capabilities (like anti-aliasing) are currently available through the high-level function.


Interactive Jupyter Viewing
===========================

The rendering function is fully compatible with :ref:`Kaolin Interactive 3D Viewer <visualizer>`, enabling easy inspection of rendering passes
right in your debugging notebook:

.. raw:: html

   <video width="456" height="360" autoplay="true" loop="true" controls>
   <source src="../_static/visualizer_urchin.mp4" type="video/mp4">
   </video>
   <div style="font-size: 75%; margin-bottom: 20px;">
   Interactively view/debug the output of any custom render function in a Jupyter notebook <br/>
   (here showing differentiable rendering of a <a src="https://sketchfab.com/models/c5c2ad0175f943969abc4d2368c0d2ff/embed">Sea Urchin Shell</a> by <a href="https://sketchfab.com/drakery">Drakery</a> from <a href="https://www.sketchfab.com/">Sketchfab</a>).
   </div>
