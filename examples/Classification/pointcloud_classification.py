import argparse
import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count

from kaolin.datasets import ModelNet
from kaolin.models.PointNet import PointNetClassifier
import kaolin.transforms as tfs

parser = argparse.ArgumentParser()
parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
parser.add_argument('--categories', type=str, nargs='+', default=['chair', 'sofa'], help='list of object classes to use.')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points to sample from meshes.')
parser.add_argument('--epochs', type=int, default=10, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=12, help='Batch size.')
parser.add_argument('--viz-test', action='store_true', help='Visualize an output of a test sample')
parser.add_argument('--transforms-device', type=str, default='cuda', help='Device to use.')

args = parser.parse_args()


def to_device(inp):
    inp.to(args.transforms_device)
    return inp

transform = tfs.Compose([
    to_device,
    tfs.TriangleMeshToPointCloud(num_samples=args.num_points),
    tfs.NormalizePointCloud()
])

if args.transforms_device == 'cuda':
    num_workers = 0
    pin_memory = False
else:
    num_workers = cpu_count()
    pin_memory = True

train_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
                                   split='train', transform=transform),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)

val_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
                                 split='test', transform=transform),
                        batch_size=args.batch_size,
                        num_workers=num_workers, pin_memory=pin_memory)

model = PointNetClassifier(num_classes=len(args.categories)).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()
start_time = time.time()
for e in range(args.epochs):

    print('###################')
    print('Epoch:', e)
    print('###################')

    train_loss = 0.
    train_accuracy = 0.
    num_batches = 0

    model.train()

    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(train_loader)):
        category = batch['attributes']['category'].cuda()
        pred = model(batch['data'].cuda())
        loss = criterion(pred, category.view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred_label = torch.argmax(pred, dim=1)
        train_accuracy += torch.mean((pred_label == category.view(-1)).float()).detach().cpu().item()
        num_batches += 1

    print('Train loss:', train_loss / num_batches)
    print('Train accuracy:', train_accuracy / num_batches)

    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            category = batch['attributes']['category'].cuda()
            pred = model(batch['data'].cuda())
            loss = criterion(pred, category.view(-1))
            val_loss += loss.item()

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            val_accuracy += torch.mean((pred_label == category.view(-1)).float()).cpu().item()
            num_batches += 1

    print('Val loss:', val_loss / num_batches)
    print('Val accuracy:', val_accuracy / num_batches)
end_time = time.time()
print('Training time: {}'.format(end_time - start_time))
test_loader = DataLoader(ModelNet(args.modelnet_root, categories=args.categories,
                                  split='test', transform=transform),
                         shuffle=True, batch_size=15, num_workers=num_workers, pin_memory=pin_memory)

test_batch = next(iter(test_loader))
preds = model(test_batch['data'].cuda())
pred_labels = torch.max(preds, axis=1)[1]

if args.viz_test:
    from utils import visualize_batch
    visualize_batch(test_batch, pred_labels, labels, args.categories)
