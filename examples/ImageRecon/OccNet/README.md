# Example: OccNet
This example allows you to train and test the Occupancy Networks algorithm for single image 3D object reconstruction. For details on the algorithms see: "Occupancy Networks: Learning 3D Reconstruction in Function Space": https://arxiv.org/abs/1812.03828.

Note: `cache-dir` specifies the location to which intermediate asset
representations will be stored

### Training the network:

To train the system call
```bash
python train.py \
    --shapenet-root <path/to/ShapeNet> \
    --shapenet-images-root <path/to/ShapeNetImages> \
    --categories <category list> \
    --cache-dir /cache/
```


### Evaluating the network: 

To evaluate a trained model
```bash
python eval.py \
    --shapenet-root <path/to/ShapeNetImages> \
    --shapenet-images-root <path/to/ShapeNetImages> \
    --categories <category list> \
    --cache-dir /cache/
```
