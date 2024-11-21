# Example: AtlasNet
This example includes training and testing of the AtlasNet algorithm for 
single image 3D object reconstruction. For details on the algorithms see: 
["AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation"](https://arxiv.org/abs/1802.05384)

Note: `cache-dir` specifies the location to which intermediate asset
representations will be stored.


### Training the network:

#### Option 1 : Without auto-encoder pre-training (faster to train, poorer performances)
```bash
python train.py --shapenet-root <path_to_ShapeNetCore.v1>  \
	--shapenet-images-root <path_to_ShapeNetRendering> \
	--expid AtlasNet_SVR
```

Data : [Pointclouds - ShapeNetv1](https://www.shapenet.org/) - [Renderings](http://3d-r2n2.stanford.edu/)

#### Option 2 : With auto-encoder pre-training (better results)

```bash
python train_auto_encoder.py --shapenet-root <path_to_ShapeNetCore.v1>  \
	--expid AtlasNet_AE
```
This will train the point auto-encoder. Then call the following to train the image 
reconstruction algorithm: 

```bash
python train.py --shapenet-root <path_to_ShapeNetCore.v1>  \
	--shapenet-images-root PATH_TO_ShapeNetRendering  \
	--expid AtlasNet_SVR  \
	--expid-decoder AtlasNet_AE
```

### Evaluating the network: 
Download pre-trained models [here](https://drive.google.com/a/polytechnique.org/uc?id=1gqOYpIyvqUohECWom8bUgWRzClJNwOjK&export=download).

(Note, in the Trimesh window, tap `c` to disable backface culling)

```bash
# To evaluate the auto-encoder
python eval_auto_encoder.py --shapenet-root <path_to_ShapeNetCore.v1> --expid AtlasNet_AE
# Output
Chamfer Loss over validation set is 0.000838079037599338
F-score over validation set is 0.23783030159928953
```
```bash
# To evaluate a trained model
python eval.py --shapenet-root <path_to_ShapeNetCore.v1> \
    --shapenet-images-root <path_to_ShapeNetRendering> --expid AtlasNet_SVR
# Output
Chamfer Loss over validation set is 0.003955722602290667
F-score over validation set is 0.09235857071979071
```

