import torch
from torch.utils.data import DataLoader
import kaolin as kal
from kaolin import ClassificationEngine
from kaolin.datasets import ModelNet10
from kaolin.models.PointNet import PointNetClassifier as PointNet
from kaolin.transforms import NormalizePointCloud

norm = NormalizePointCloud()
train_loader = DataLoader(ModelNet10('/path/to/ModelNet10', categories=['chair', 'sofa'],
                                     split='train', rep='pointcloud', transform=norm, device='cuda:0'),
                          batch_size=12, shuffle=True)
val_loader = DataLoader(ModelNet10('/path/to/ModelNet10', categories=['chair', 'sofa'],
                                   split='test', rep='pointcloud', transform=norm, device='cuda:0'),
                        batch_size=12)
engine = ClassificationEngine(PointNet(num_classes=2), train_loader, val_loader, device='cuda:0')
engine.fit()
