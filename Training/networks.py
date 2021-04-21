
import torch
import torch.nn as nn
import torch.nn.functional as F



class SuperPointNet(nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        #         self.conv1a.weight.register_hook(lambda grad: grad * 0)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        #         self.conv1b.weight.register_hook(lambda grad: grad * 0)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        #         self.conv2a.weight.register_hook(lambda grad: grad * 0)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        #         self.conv2b.weight.register_hook(lambda grad: grad * 0)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        #         self.conv3a.weight.register_hook(lambda grad: grad * 0)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        #         self.conv3b.weight.register_hook(lambda grad: grad * 0)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        #         self.conv4a.weight.register_hook(lambda grad: grad * 0)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        #         self.conv4b.weight.register_hook(lambda grad: grad * 0)
        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        #         self.convPa.weight.register_hook(lambda grad: grad * 0)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        #         self.convPb.weight.register_hook(lambda grad: grad * 0)
        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        #         self.convDa.weight.register_hook(lambda grad: grad * 0)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        #         self.convDb.weight.register_hook(lambda grad: grad * 0)

    def forward(self, x):
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class CNNet(nn.Module):
    '''
    Reimplementation of the network "Learning to find good correspondences"

    '''

    def __init__(self, blocks):
        '''
        Constructor.
        '''
        super(CNNet, self).__init__()

        # network takes 5 inputs per correspondence: 2D point in img1, 2D point in img2, and 1D side information like a matching ratio
        self.p_in = nn.Conv2d(5, 128, 1, 1, 0)

        self.res_blocks = []

        for i in range(0, blocks):
            self.res_blocks.append((
                nn.Conv2d(128, 128, 1, 1, 0),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 1, 1, 0),
                nn.BatchNorm2d(128),
            ))

        for i, r in enumerate(self.res_blocks):
            super(CNNet, self).add_module(str(i) + 's0', r[0])
            super(CNNet, self).add_module(str(i) + 's1', r[1])
            super(CNNet, self).add_module(str(i) + 's2', r[2])
            super(CNNet, self).add_module(str(i) + 's3', r[3])

        self.p_out = nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, x):
        '''
        Forward pass.

        inputs -- 4D data tensor (BxCxHxW)
        C -> 5 values
        h -> number of correspondences
        w -> 1 (dummy dimension)
        B -> batch size (multiple image pairs)
        '''

        x = F.relu(self.p_in(x))

        for r in self.res_blocks:
            res = x
            x = F.relu(r[1](F.instance_norm(r[0](x))))
            x = F.relu(r[3](F.instance_norm(r[2](x))))
            x = x + res

        x = self.p_out(x)
        return x




