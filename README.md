# Kaolin: A Pytorch Library for Accelerating 3D Deep Learning Research

<p align="center">
    <img src="assets/kaolin.png">
</p>

## Overview
NVIDIA Kaolin library provides a PyTorch API for working with a variety of 3D representations and includes a growing collection of GPU-optimized operations such as modular differentiable rendering, fast conversions between representations, data loading, 3D checkpoints, differentiable camera API, differentiable lighting with spherical harmonics and spherical gaussians, powerful quadtree acceleration structure called Structured Point Clouds, interactive 3D visualizer for jupyter notebooks, convenient batched mesh container and more. Visit the [Kaolin Library Documentation](https://kaolin.readthedocs.io/en/latest/) to get started!

Note that Kaolin library is part of the larger [NVIDIA Kaolin effort](https://developer.nvidia.com/kaolin) for 3D deep learning.

## Installation and Getting Started

Starting with v0.12.0, Kaolin supports installation with wheels:
```
# Replace TORCH_VERSION and CUDA_VERSION with your torch / cuda versions
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VERSION}_cu{CUDA_VERSION}.html
```
For example, to install kaolin 0.15.0 over torch 1.12.1 and cuda 11.3:
```
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
```

## About the Latest Release (0.15.0)

In this version we added a [non commercial section](https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html) under [NSCL license](LICENSE.NSCL). See [The license section for more info](#Licenses) for more details.

In this new section we implemented [features for Flexicubes](https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html#kaolin.non_commercial.FlexiCubes) a method to extract meshes from scalar fields. See more information in [the official repository](https://github.com/nv-tlabs/FlexiCubes) which is now using Kaolin's implementation.

<a href="https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html#kaolin.non_commercial.FlexiCubes"><img src="./assets/flexicubes.png" alt="flexicubes" height="250" /></a>

In addition we implemented a [GLTF mesh loader](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.gltf.html) that can be used to load models from [Objaverse](https://objaverse.allenai.org/objaverse-1.0) and [Objaverse-XL](https://objaverse.allenai.org/).

<a href="https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.gltf.html"><img src="./assets/gltf.png" alt="gltf" height="250" /></a>


Check our new tutorial:
[**Load and render a GLTF file** interactively into a Jupyter notebook:](https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/gltf_viz.ipynb)
In this file we show how to load a gltf file and fully differentiably render it with [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) and [spherical gaussian for diffuse and specular lighting](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.lighting.html), using displacement mapping and other materials properties from the GLTF file.

<a href="https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/gltf_viz.ipynb"><img src="./assets/avocado.png" alt="gltf notebook" height="250" /></a>

See [change logs](https://github.com/NVIDIAGameWorks/kaolin/releases/tag/v0.15.0) for details.

## Contributing

Please review our [contribution guidelines](CONTRIBUTING.md).

## External Projects using Kaolin

* [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp):
   * Use [Camera API](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.camera.html), [Structured Point Clouds](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html) and its [rendering capabilities](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.spc.html)
* [gradSim: Differentiable simulation for system identification and visuomotor control](https://github.com/gradsim/gradsim):
   * Use [DIB-R rasterizer](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.dibr_rasterization), [obj loader](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html#kaolin.io.obj.import_mesh) and [timelapse](https://kaolin.readthedocs.io/en/latest/modules/kaolin.visualize.html#kaolin.visualize.Timelapse)
* [Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer](https://github.com/nv-tlabs/DIB-R-Single-Image-3D-Reconstruction/tree/2cfa689881145c8e0647ae8dd077e55b5a578658):
   * Use [Kaolin's DIB-R rasterizer](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.dibr_rasterization), [camera functions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.camera.html) and [Timelapse](https://kaolin.readthedocs.io/en/latest/modules/kaolin.visualize.html#kaolin.visualize.Timelapse) for 3D checkpoints.
* [Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces](https://github.com/nv-tlabs/nglod):
    * Use [SPC](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html) conversions and [ray-tracing](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.spc.html#kaolin.render.spc.unbatched_raytrace), yielding 30x memory and 3x training time reduction.
* [Learning Deformable Tetrahedral Meshes for 3D Reconstruction](https://github.com/nv-tlabs/DefTet):
    * Use [Kaolin's DefTet volumetric renderer](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#kaolin.render.mesh.deftet_sparse_render), [tetrahedral losses](https://kaolin.readthedocs.io/en/latest/modules/kaolin.metrics.tetmesh.html), [camera_functions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.camera.html), [mesh operators and conversions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.html), [ShapeNet dataset](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.shapenet.html#kaolin.io.shapenet.ShapeNetV1), [point_to_mesh_distance](https://kaolin.readthedocs.io/en/latest/modules/kaolin.metrics.trianglemesh.html#kaolin.metrics.trianglemesh.point_to_mesh_distance) and [sided_distance](https://kaolin.readthedocs.io/en/latest/modules/kaolin.metrics.pointcloud.html#kaolin.metrics.pointcloud.sided_distance).
* [Text2Mesh](https://github.com/threedle/text2mesh):
    * Use [Kaolin's rendering functions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.mesh.html#), [camera functions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.render.camera.html), and [obj](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html#kaolin.io.obj.import_mesh) and [off](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.off.html#kaolin.io.off.import_mesh) importers.
* [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (FlexiCubes)
](https://github.com/nv-tlabs/FlexiCubes):
    * Use [Flexicube class](https://kaolin.readthedocs.io/en/latest/modules/kaolin.non_commercial.html#kaolin.non_commercial.FlexiCubes), [obj loader](https://kaolin.readthedocs.io/en/latest/modules/kaolin.io.obj.html), [turntable visualizer](https://kaolin.readthedocs.io/en/latest/modules/kaolin.visualize.html#kaolin.visualize.IpyTurntableVisualizer)

## Licenses

Most of Kaolin's repository is under [Apache v2.0 license](LICENSE), except under [kaolin/non_commercial](kaolin/non_commercial/) which is under [NSCL license](LICENSE.NSCL) restricted to non commercial usage for research and evaluation purposes. For example, FlexiCubes method is included under [non_commercial](kaolin/non_commercial/flexicubes/flexicubes.py).

Default `kaolin` import includes Apache-licensed components:
```
import kaolin
```

The non-commercial components need to be explicitly imported as:
```
import kaolin.non_commercial
```

## Citation

If you are using Kaolin library for your research, please cite:

```
@misc{KaolinLibrary,
      author = {Fuji Tsang, Clement and Shugrina, Maria and Lafleche, Jean Francois and Takikawa, Towaki and Wang, Jiehan and Loop, Charles and Chen, Wenzheng and Jatavallabhula, Krishna Murthy and Smith, Edward and Rozantsev, Artem and Perel, Or and Shen, Tianchang and Gao, Jun and Fidler, Sanja and State, Gavriel and Gorski, Jason and Xiang, Tommy and Li, Jianing and Li, Michael and Lebaredian, Rev},
      title = {Kaolin: A Pytorch Library for Accelerating 3D Deep Learning Research},
      year = {2022},
      howpublished={\url{https://github.com/NVIDIAGameWorks/kaolin}}
}
```

## Contributors

Current Team:

- Technical Lead: Clement Fuji Tsang
- Manager: Maria (Masha) Shugrina
- Charles Loop
- Or Perel
- Alexander Zook

Other Majors Contributors:

- Wenzheng Chen
- Sanja Fidler
- Jun Gao
- Jason Gorski
- Jean-Francois Lafleche
- Rev Lebaredian
- Jianing Li
- Michael Li
- Krishna Murthy Jatavallabhula
- Artem Rozantsev
- Tianchang (Frank) Shen
- Edward Smith
- Gavriel State
- Towaki Takikawa
- Jiehan Wang
- Tommy Xiang
