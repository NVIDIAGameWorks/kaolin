import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import kaolin as kal


epochs = 10
lr = 1e-3
device = 'cuda:0'
normpc = kal.transforms.NormalizePointCloud()
data = kal.datasets.ModelNet10('/path/to/ModelNet10',
                               categories=['bed', 'bathtub'],
                               split='train', rep='pointcloud',
                               transform=normpc, device=device)
loader = DataLoader(data, batch_size=12, shuffle=True)
val_data = kal.datasets.ModelNet10('/path/to/ModelNet10',
                               categories=['bed', 'bathtub'],
                               split='test', rep='pointcloud',
                               transform=normpc, device=device)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
model = kal.models.PointNet.PointNetClassifier(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

for e in range(epochs):

    print('###################')
    print('Epoch:', e)
    print('###################')

    train_loss = 0.
    train_accuracy = 0.
    num_batches = 0

    model.train()

    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(loader)):
        pred = model(batch[0])
        loss = criterion(pred, batch[1].view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred_label = torch.argmax(pred, dim=1)
        train_accuracy += torch.mean((pred_label == batch[1].view(-1)).float(
            )).detach().cpu().item()
        num_batches += 1

    print('Train loss:', train_loss / num_batches)
    print('Train accuracy:', train_accuracy / num_batches)

    val_loss = 0.
    val_accuracy = 0.
    num_batches = 0

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            pred = model(batch[0])
            loss = criterion(pred, batch[1].view(-1))
            val_loss += loss.item()

            # Compute accuracy
            pred_label = torch.argmax(pred, dim=1)
            val_accuracy += torch.mean((pred_label == batch[1].view(-1)).float(
                )).cpu().item()
            num_batches += 1

    print('Val loss:', val_loss / num_batches)
    print('Val accuracy:', val_accuracy / num_batches)
