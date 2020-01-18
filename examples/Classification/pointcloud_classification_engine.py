import argparse

import torch
from torch.utils.data import DataLoader

import kaolin as kal
import kaolin.transforms as tfs
from kaolin import ClassificationEngine
from kaolin.datasets import ModelNet
from kaolin.models.PointNet import PointNetClassifier
from kaolin.transforms import NormalizePointCloud


parser = argparse.ArgumentParser()
parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
parser.add_argument('--categories', type=str, nargs='+', default=['chair', 'sofa'], help='list of object classes to use.')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points to sample from meshes.')
parser.add_argument('--epochs', type=int, default=10, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=12, help='Batch size.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')

args = parser.parse_args()

assert len(args.categories) >= 2, 'At least two categories must be specified.'

transform = tfs.Compose([
    tfs.TriangleMeshToPointCloud(num_samples=args.num_points),
    tfs.NormalizePointCloud()
])

train_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
                                   split='train', transform=transform, device=args.device),
                          batch_size=args.batch_size, shuffle=True)

val_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
                                 split='test',transform=transform, device=args.device),
                        batch_size=args.batch_size)

model = PointNetClassifier(num_classes=len(args.categories))
engine = ClassificationEngine(model, train_loader, val_loader, device=args.device)
engine.fit()
