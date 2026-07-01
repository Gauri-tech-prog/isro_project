import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        return torch.cat((self.model(x), skip), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.d1 = UNetDown(in_channels, 64, normalize=False)
        self.d2 = UNetDown(64, 128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512, dropout=0.5)
        self.d5 = UNetDown(512, 512, dropout=0.5)
        self.d6 = UNetDown(512, 512, dropout=0.5)
        self.d7 = UNetDown(512, 512, dropout=0.5)
        self.d8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.u1 = UNetUp(512,  512, dropout=0.5)
        self.u2 = UNetUp(1024, 512, dropout=0.5)
        self.u3 = UNetUp(1024, 512, dropout=0.5)
        self.u4 = UNetUp(1024, 512, dropout=0.5)
        self.u5 = UNetUp(1024, 256)
        self.u6 = UNetUp(512,  128)
        self.u7 = UNetUp(256,  64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1=self.d1(x); d2=self.d2(d1); d3=self.d3(d2); d4=self.d4(d3)
        d5=self.d5(d4); d6=self.d6(d5); d7=self.d7(d6); d8=self.d8(d7)
        u1=self.u1(d8,d7); u2=self.u2(u1,d6); u3=self.u3(u2,d5)
        u4=self.u4(u3,d4); u5=self.u5(u4,d3); u6=self.u6(u5,d2)
        u7=self.u7(u6,d1)
        return self.final(u7)