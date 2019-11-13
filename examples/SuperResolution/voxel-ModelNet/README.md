# Example: voxel-superresolution

These example scripts lets you train a voxel superresolution network on the ModelNet Dataset using a simple encoder-decoder network. There are two training schemes, one using MSE loss and the other using negative log likelihood loss. 


### Training the network: MSE

To train using MSE run
```
python train_MSE.py
```


### Evaluating the network: MSE

To evaluate a trained MSE model run 
```
python eval_MSE.py
```


### Training the network: NLLL

To train using NLLL run
```
python train_NLLL.py
```


### Evaluating the network: NLLL

To evaluate a trained NLLL model run 
```
python eval_NLLL.py
```

