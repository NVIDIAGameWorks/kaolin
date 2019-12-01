# Example: voxel-superresolution

These example scripts let you train a voxel superresolution network on the ShapeNet Dataset using a simple encoder-decoder network. The task for the network is to take a low-resolution voxelized object as input and output a high resolution voxelized object.


### Training the network

To train:
```
python train.py --shapenet-root <ShapeNet dir> --cache-dir 'cache/'
```


### Evaluating the network

To evaluate:
```
python eval.py --shapenet-root <ShapeNet dir> --cache-dir 'cache/'
```
