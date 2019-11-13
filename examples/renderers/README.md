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

## Building

Before running the examples in the `test` directory, ensure that you step into the `DIB-R`, the `NMR`, and the `SoftRas` directories respectively, and install the renderers following the steps outlined in their respective `README` files.


## Testing

Then, navigate into the `test` directory, and run any of the tests. For example, test Neural Mesh Renderer by running

```
python test_nmr.py
```

or, for SoftRasterizer

```
python test_softras.py
```

or, for DIB-Renderer
```
python test_dibr.py
```
