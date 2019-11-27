# Example: GeoMetrics
This example includes training and testing of the GEOMetrics algorithm for 
single image 3D object reconstruction. For details on the algorithms see: 
"GEOMetrics: Exploiting Geometric Structure for Graph-Encoded Objects": 
https://arxiv.org/abs/1901.11461

Training on a single class takes several days using an NVIDIA RTX 2080Ti
GPU. To obtain more visually pleasing results at the expense of F-Score,  
increase the regularizer weights (laplace, edge).

Note: `cache-dir` specifies the location to which intermediate asset
representations will be stored.


### Training the network:

#### Option 1 : No latent embedding loss (faster to train)
```bash
python train.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-images-root <path/to/ShapeNetImages> \
    --categories <category list> \
    --cache-dir cache/
```


#### Option 2 : With latent embedding loss (better results)
```bash
python train_auto_encoder.py --shapenet-root <path/to/ShapeNet> \
    --cache-dir cache/
```
This will train the mesh encoder. Then call the following to train the image 
reconstruction algorithm: 
```bash
python train.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-images-root <path/to/ShapeNetImages> \
    --categories <category list> \
    --cache-dir cache/ --latent_loss
```

### Evaluating the network: 
To evaluate the auto-encoder
```bash
python eval_auto_encoder.py --shapenet-root <path/to/ShapeNet> \
    --categories <category list> \
    --cache-dir cache/
```
To evaluate a trained model
(Note, in the Trimesh window, tap `c` to disable backface culling)
```bash
python eval.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-images-root <path/to/ShapeNetImages> \
    --categories <category list> \
    --cache-dir cache/
```
