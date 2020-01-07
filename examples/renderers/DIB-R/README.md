# DIB-Renderer Example

![example image with vertex color](vertex-color.gif)
![example image with texture](textured.gif)

This example uses the DIB-R renderer in kaolin to render a simple mesh, and output an animated GIF image of the rendered object.

Usage:

To render the mesh using vertex position as vertex color:

```bash
python example.py
```

To render a textured mesh:

```bash
python example.py --use_texture
```

By default, the example will render the `banana.obj` mesh in this directory, and output the result into a `results` folder in this directory.

If `--use_texture` is specified, the example will use the `texture.png` image in this directory as texture. To use your own texture, add the option `--texture <path-to-texture>`.

Use the `--help` option to display available arguments.

The DIB-R renderer is based on its original implementation by Wenzheng et al.

```
@inproceedings{chen2019dibrender,
title={Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
author={Wenzheng Chen and Jun Gao and Huan Ling and Edward Smith and Jaakko Lehtinen and Alec Jacobson and Sanja Fidler},
booktitle={Advances In Neural Information Processing Systems},
year={2019}
}
```
