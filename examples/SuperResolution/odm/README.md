# Example: ODM - Super Resolution

These example scripts lets you train a voxel Super-Resolution network on the ShapeNet Dataset using Orthographic Depth Maps(ODMS). For details on the algorithms see: "Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation" : https://arxiv.org/abs/1802.09987. 

There are two training schemes, one which directly predicts superresolves ODMs (Direct), and one which separates the problem into occupancy and residual predictions (MVD). 

## Direct
### Training

```
python train.py --mode Direct \
    --shapenet-root <path/to/ShapeNet> \
    --categories <category list> \
    --cache-dir <cache dir>
```


### Evaluating

```
python eval.py --mode Direct \
    --shapenet-root <path/to/ShapeNet> \
    --categories <category list> \
    --cache-dir <cache dir>
```

## MVD
### Training

```
python train.py --mode MVD \
    --shapenet-root <path/to/ShapeNet> \
    --categories <category list> \
    --cache-dir <cache dir>
```


### Evaluating

```
python eval.py --mode MVD \
    --shapenet-root <path/to/ShapeNet> \
    --categories <category list> \
    --cache-dir <cache dir>
```
