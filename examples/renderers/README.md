# Differentiable renderers

This set of examples demonstrates how implementations of popular differentiable renderers:

1. Neural Mesh Renderer
2. Soft Rasterizer
3. DIB-Renderer

We thank the authors of the above papers for making their code publicly available. If you use any part of the code from the respective folders, consider citing the authors' original publications:

Neural mesh renderer
```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```

Soft Rasterizer
```
@article{liu2019softras,
  title={Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning},
  author={Liu, Shichen and Li, Tianye and Chen, Weikai and Li, Hao},
  journal={The IEEE International Conference on Computer Vision (ICCV)},
  month = {Oct},
  year={2019}
}
```

DIB-Renderer
```
@inproceedings{chen2019dibrender,
  title={Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
  author={Wenzheng Chen and Jun Gao and Huan Ling and Edward Smith and Jaakko Lehtinen and Alec Jacobson and Sanja Fidler},
  booktitle={Advances In Neural Information Processing Systems},
  year={2019}
}
```

# SoftRasterizer

This popular renderer is based on the 2019 ICCV paper by Schichen Liu, Tianye Li, Weikai Chen, and Hao Li. Softras is a differentiable _rasterization-based_ renderer that can provide gradient with respect to the vertex positions and texture (color) of a Mesh. The following examples demonstrate a typical use cases, and can be repurposed to suit your needs.

Refer to our [examples](softras) for details on how to use this differentiable rendering module for inverse graphics tasks.
