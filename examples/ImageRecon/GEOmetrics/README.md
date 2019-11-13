# Example: GeoMetrics
This example allows you to train and test the GEOMetrice algorithm for signle image 3D object reconstruction. For details on the algorithms see: "GEOMetrics: Exploiting Geometric Structure for Graph-Encoded Objects" : https://arxiv.org/abs/1901.11461


### Training the network:

To train the system call
```
python train.py
```


To train the system with the latent encoding loss call first: 
```
python train_auto_encoder.py
```
This will train the mesh encoder. Then call the follwoing to train the image reconstruction algorithm: 
```
python train.py -latent_loss
```

### Evaluating the network: 

To evaluate a trained model
```
python eval.py
```




