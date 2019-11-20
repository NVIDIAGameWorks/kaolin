"""
Network architecture definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderDecoder_32_128(nn.Module):
    def __init__(self):
        super(EncoderDecoder_32_128, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride = 2, padding=0), 
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride = 2, padding=1), 
            nn.BatchNorm3d(32),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(8), 
            nn.ReLU(),
            nn.ConvTranspose3d(8, 4, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(4), 
            nn.ReLU(),
            nn.ConvTranspose3d(4, 1, 3, stride = 2,  padding=0), 
        
            )

    
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        

        return torch.sigmoid(x)[:,0,:128, :128, :128]




class EncoderDecoderForNLL_32_128(nn.Module):
    def __init__(self):
        super(EncoderDecoderForNLL_32_128, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride = 2, padding=1), 
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride = 2, padding=1), 
            nn.BatchNorm3d(32),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(8), 
            nn.ReLU(),
            nn.ConvTranspose3d(8, 4, 3, stride = 2,  padding=0), 
            nn.BatchNorm3d(4), 
            nn.ReLU(),
            nn.ConvTranspose3d(4, 2, 3, stride = 2,  padding=0), 
            )

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)[:,:,:128, :128, :128]

        return torch.exp(self.log_softmax(x))
