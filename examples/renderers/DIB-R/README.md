## DIB-Renderer (DIB-R)

This popular renderer is based on the 2019 NeurIPS paper by Wenzheng Chen, Jun Gao, Huan Ling, Edward Smith, Jaakko Lehtinen, Alec Jacobson, and Sanja Fidler. DIB-R is a differentiable _rasterization-based_ renderer that can provide gradient with respect to the vertex positions and texture (color as well as _uv_ maps) of a Mesh. Further, it can support a variety of shading modes (Lambertian, Phong, Spherical Harmonics). The following examples demonstrate a typical use cases, and can be repurposed to suit your needs.

### Example 1: Simple rendering

The [`dibr_simple_render`](dibr_simple_render.py) script provides a breezy intro to the renderer API. To run this example, execute the following command

```bash
python dibr_simple_render.py
```

This should render a _banana_ from multiple views, and produce a result that looks like this:
<p align="center">
  <img src="assets/dibr_rendered_vertexcolor.gif">
</p>

If run with the `--use-texture` flag, it wraps a checkerboard texture image around the rendered banana.
```bash
python dibr_simple_render.py --use-texture
```

You should see something similar to this:
<p align="center">
  <img src="assets/dibr_rendered_texture.gif">
</p>


### Example 2: Vertex optimization

The [`dibr_vertex_optimization`](dibr_vertex_optimization.py) script provides a simple example that demonstrates the usage of gradients with respect to vertex positions. We begin with a sphere mesh, and iteratively deform the mesh until the image formed resembles the banana image rendered in the previous example. Execute

```bash
python dibr_vertex_optimization.py
```

This should generate two `gif` files showing the optimization output and the resultant mesh respectively.
<p align="center">
  <img src="assets/dibr_vertex_optimization_progress.gif">
</p>
<p align="center">
  <img src="assets/dibr_vertex_optimization_output.gif">
</p>

### Example 3: Texture optimization

The [`dibr_texture_optimization`](dibr_texture_optimization.py) script provides a simple example that demonstrates the usage of gradients with respect to texture (face colors). We begin with a banana mesh, and want to infer the color of the mesh by looking at the image rendered in the first example. We iteratively update the texture of the mesh until the image formed resembles the banana image rendered in the first example. Execute

```bash
python dibr_texture_optimization.py
```

This should generate two `gif` files showing the optimization output and the resultant mesh respectively.
<p align="center">
  <img src="assets/dibr_texture_optimization_progress.gif">
</p>
<p align="center">
  <img src="assets/dibr_texture_optimization_output.gif">
</p>

The DIB-Renderer is based on its original implementation by Wenzheng et al.

```
@inproceedings{chen2019dibrender,
title={Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
author={Wenzheng Chen and Jun Gao and Huan Ling and Edward Smith and Jaakko Lehtinen and Alec Jacobson and Sanja Fidler},
booktitle={Advances In Neural Information Processing Systems},
year={2019}
}
```
