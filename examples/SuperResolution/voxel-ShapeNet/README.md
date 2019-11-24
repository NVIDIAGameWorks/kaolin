# Example: voxel-superresolution

These example scripts let you train a voxel superresolution network on the ShapeNet Dataset using a simple encoder-decoder network. There are two training schemes, one using MSE loss and the other using negative log likelihood loss. 


### Training the network: MSE

To train using MSE run
```
python train.py --shapenet-root <ShapeNet dir> --cache <cache dir> --loss-type MSE
```


### Evaluating the network: MSE

To evaluate a trained MSE model run 
```
python eval.py --shapenet-root <ShapeNet dir> --cache <cache dir> --loss-type MSE
```

### Training the network: NLLL

To train using NLLL run
```
python train.py --shapenet-root <ShapeNet dir> --cache <cache dir> --loss-type MSE
```


### Evaluating the network: NLLL

To evaluate a trained NLLL model run 
```
python eval.py --shapenet-root <ShapeNet dir> --cache <cache dir> --loss-type MSE
```
