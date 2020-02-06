import torch 
from torch import nn
import torch.nn.functional as F

from .SimpleGCN import SimpleGCN

class MeshEncoder(nn.Module):
    def __init__(self, latent_length):
        super(MeshEncoder, self).__init__()
        self.h1 = SimpleGCN(3, 60)
        self.h21 = SimpleGCN(60, 60)
        self.h22 = SimpleGCN(60, 60)
        self.h23 = SimpleGCN(60, 60)
        self.h24 = SimpleGCN(60, 120)
        self.h3 = SimpleGCN(120, 120)
        self.h4 = SimpleGCN(120, 120)
        self.h41 = SimpleGCN(120, 150)
        self.h5 = SimpleGCN(150, 200)
        self.h6 = SimpleGCN(200, 210)
        self.h7 = SimpleGCN(210, 250)
        self.h8 = SimpleGCN(250, 300)
        self.h81 = SimpleGCN(300, 300)
        self.h9 = SimpleGCN(300, 300)
        self.h10 = SimpleGCN(300, 300)
        self.h11 = SimpleGCN(300, 300)
        self.reduce = SimpleGCN(300, latent_length) 

    def resnet(self, features, res):
        temp = features[:, :res.shape[1]]
        temp = temp + res
        features = torch.cat((temp, features[:, res.shape[1]:]), dim=1)
        return features, features

    def forward(self, positions, adj):
        res = positions
        features = F.elu(self.h1(positions, adj))
        features = F.elu(self.h21(features, adj))
        features = F.elu(self.h22(features, adj))
        features = F.elu(self.h23(features, adj))
        features = F.elu(self.h24(features, adj))
        features = F.elu(self.h3(features, adj))
        features = F.elu(self.h4(features, adj))
        features = F.elu(self.h41(features, adj))
        features = F.elu(self.h5(features, adj))
        features = F.elu(self.h6(features, adj))
        features = F.elu(self.h7(features, adj))
        features = F.elu(self.h8(features, adj))
        features = F.elu(self.h81(features, adj))
        features = F.elu(self.h9(features, adj))
        features = F.elu(self.h10(features, adj))
        features = F.elu(self.h11(features, adj))

        latent = F.elu(self.reduce(features , adj))  
        latent = (torch.max(latent, dim=0)[0])      
        return latent
