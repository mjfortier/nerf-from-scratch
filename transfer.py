import numpy as np
import torch
from torch import nn

class TransferFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Linear(316, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.layers3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )
        self.layers4 = nn.Sequential(
            nn.Linear(280, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    # Takes a position (x,y,z) and an orientation(theta, phi)
    # outputs (r,g,b)
    def forward(self, pos, ori):
        # Encode inputs
        enc_pos = self.encode_position(pos)
        enc_ori = self.encode_orientation(ori)
        
        #Start forward pass
        op1 = self.layers1(enc_pos)

        # Add enc_pos again partway through
        l2_input = torch.cat((enc_pos, op1), 0)
        op2 = self.layers2(l2_input)

        sigma = self.layers3(op2)

        l4_input = torch.cat((enc_ori, op2), 0)
        rgb = self.layers4(l4_input)

        return torch.cat((sigma, rgb), 0)

    # Input: (3,) orientation vector
    # Output: (60,) encoded vector
    def encode_position(self, pos):
        enc_pos = torch.zeros(60)
        for i in range(10):
            enc_pos[6*i + 0] = torch.sin(2**i*np.pi*pos[0])
            enc_pos[6*i + 1] = torch.cos(2**i*np.pi*pos[0])
            enc_pos[6*i + 2] = torch.sin(2**i*np.pi*pos[1])
            enc_pos[6*i + 3] = torch.cos(2**i*np.pi*pos[1])
            enc_pos[6*i + 4] = torch.sin(2**i*np.pi*pos[2])
            enc_pos[6*i + 5] = torch.cos(2**i*np.pi*pos[2]) 
        return enc_pos

    # Input: (3,) orientation vector
    # Output: (24,) encoded vector
    def encode_orientation(self, ori):
        enc_ori = torch.zeros(24)
        for i in range(4):
            enc_ori[6*i + 0] = torch.sin(2**i*np.pi*ori[0])
            enc_ori[6*i + 1] = torch.cos(2**i*np.pi*ori[0])
            enc_ori[6*i + 2] = torch.sin(2**i*np.pi*ori[1])
            enc_ori[6*i + 3] = torch.cos(2**i*np.pi*ori[1])
            enc_ori[6*i + 4] = torch.sin(2**i*np.pi*ori[2])
            enc_ori[6*i + 5] = torch.cos(2**i*np.pi*ori[2])
        return enc_ori

    #TODO: implement backprop
    def backward(self):
        pass       

