# Example: ODM-superresolution 

These example scripts lets you train a voxel superresolution network on the ModelNet Dataset using a Orthographic Deppth Maps(ODMS). For details on the algorithms see: "Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation" : https://arxiv.org/abs/1802.09987. There are two training schemes, one using which Directly predicts superresolves ODMs (Direct), and one which seperates the problem into rccupancy and residual predictions (MVD). 


### Training the network: Direct

To train using Direct run
```
python train_Direct.py
```


### Evaluating the network: Direct

To evaluate a trained Direct model run 
```
python eval_Direct.py
```


### Training the network: MVD

To train using MVD run
```
python train_MVD.py
```


### Evaluating the network: MVD

To evaluate a trained MVD model run 
```
python eval_MVD.py
```

