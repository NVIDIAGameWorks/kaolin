# Example: GeoMetrics
This example allows you to train and test the GEOMetrics algorithm for 
single image 3D object reconstruction. For details on the algorithms see: 
"GEOMetrics: Exploiting Geometric Structure for Graph-Encoded Objects": 
https://arxiv.org/abs/1901.11461


### Training the network:

#### Option 1 : No latent embedding loss (faster to train)
```bash
python train.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-rendering-root <path/to/ShapeNetRendering> \
    --cache-dir cache/
```


#### Option 2 : With latent embedding loss (better results)
```
python train_auto_encoder.py --shapenet-root <path/to/ShapeNet> \
    --cache-dir cache/
```
This will train the mesh encoder. Then call the following to train the image 
reconstruction algorithm: 
```bash
python train.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-rendering-root <path/to/ShapeNetRendering> \
    --cache-dir cache/ --latent_loss
```

### Evaluating the network: 

To evaluate a trained model
```bash
python eval.py --shapenet-root <path/to/ShapeNet> \
    --shapenet-rendering-root <path/to/ShapeNetRendering> \
    --cache-dir cache/
```
