import torch
import cv2
import numpy as np 

class SquareFinder(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([0.5]))
        self.y = torch.nn.Parameter(torch.tensor([0.5]))
        self.z = torch.nn.Parameter(torch.tensor([0.5]))
        self.w = torch.nn.Parameter(torch.tensor([0.8]))
        self.h = torch.nn.Parameter(torch.tensor([0.8]))
        self.d = torch.nn.Parameter(torch.tensor([0.8]))

        ss = torch.linspace(0, 1, s)
        # x, y = torch.meshgrid([ss,ss])
        x, y, z = torch.meshgrid([ss, ss, ss])
        

        self.mesh_grid_x = x.cuda()
        self.mesh_grid_y = y.cuda()
        self.mesh_grid_z = z.cuda()
    
    def forward(self):
        s = 256

        px = self.x
        py = self.y
        pz = self.z
        w = self.w
        h = self.h
        d = self.d
        exp = 10

        x = self.mesh_grid_x
        y = self.mesh_grid_y
        z = self.mesh_grid_z

        volume = 1 - 3*(abs((x - px)/w)**exp + abs((y - py)/h)**exp + abs((z - pz)/d)**exp)
        # TODO: project 3d to 2d
        image = volume[:,:,0].clamp(0,1)
        
        # image = 1 - 2*(abs((x - px)/w)**exp + abs((y - py)/h)**exp)  # 2d version
        # image = image.clamp(0,1)

        return image
    
    def __repr__(self):
        return f'[{self.x.item()}, {self.y.item()}, {self.z.item()}] / [{self.w.item()}, {self.h.item()}, {self.d.item()}]'

# Create Tensors to hold input and outputs.
s = 256

canvas = torch.zeros([s, s])

ss = torch.linspace(0, 1, s)
x, y = torch.meshgrid([ss, ss])
exp = 10


def rotation_matrix(theta):
    """Theta in radians."""
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    return torch.Tensor([cos, -sin, sin, cos])


# construct target box
l = 0.3
r = 0.6
b = 0.3
t = 0.7
theta = torch.tensor(torch.pi * 2/3)

rot = rotation_matrix(theta)
p = torch.tensor([l,b,r,t])
box = p  # rotated bbox. need to then convert to 2d image


# discrete image from bbox coordinates
#   there's probably a more clever way to do this
canvas = torch.zeros((s,s))
xmin, ymin, xmax, ymax = torch.floor(box*255).unbind()
for i in range(int(xmin), int(xmax)):
    for j in range(int(ymin), int(ymax)):
        canvas[i, j] = 1
y = canvas.cuda()


cv2.imwrite('tmp.png', y.data.cpu().numpy()*255)

model = SquareFinder().cuda()
criterion = torch.nn.MSELoss(reduction='mean')
# criterion = torch.nn.HuberLoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

t = 0 
while True:
    t+=1

    y_pred = model()  # "model" is the bbox, so no input parameters
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 20 == 0:
        print(t, loss.item(), model)

        zeros = np.zeros((256,256,3))

        zeros[...,2] = y_pred.cpu().data.numpy()        
        zeros[...,1] = y.cpu().data.numpy()        

        cv2.imshow('image', zeros*255)

    # KEYBOARD INTERACTIONS
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print(f'Result: {model}')