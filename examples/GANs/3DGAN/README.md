# Example: 3D GAN

This example lets you train the 3D GAN method. See: https://arxiv.org/abs/1610.07584 for details. 


### Training the network: 

To train run (adding a valid path to modelnet-root)
```
python train.py --modelnet-root <path/to/modelnet/>
```
The argument `cache-dir` specifies where to store voxelized representations of 
the ModelNet meshes used during training.


### Evaluating the network: 

To evaluate run
```
python eval.py
```
