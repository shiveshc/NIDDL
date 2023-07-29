import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        return conv1, conv2, conv3


class ConvBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv1(x))


class UNet_v(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.db1 = ConvBlock(in_channels, out_channels)
        self.db2 = ConvBlock(out_channels, out_channels)
        self.db3 = ConvBlock(out_channels, out_channels)
        self.db4 = ConvBlock(out_channels, out_channels)
        self.db5 = ConvBlock(out_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.ub1 = ConvBlockTranspose(out_channels, out_channels)
        self.cb1 = ConvBlock(2*out_channels, out_channels)
        self.ub2 = ConvBlockTranspose(out_channels, out_channels)
        self.cb2 = ConvBlock(2*out_channels, out_channels)
        self.ub3 = ConvBlockTranspose(out_channels, out_channels)
        self.cb3 = ConvBlock(2*out_channels, out_channels)
        self.ub4 = ConvBlockTranspose(out_channels, out_channels)
        self.cb4 = ConvBlock(2*out_channels, out_channels)

        self.ob1 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding='same')

        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        conv1, conv2, conv3 = self.db1(x)
        
        mp1 = self.maxpool(conv3)
        conv4, conv5, conv6 = self.db2(mp1)

        mp2 = self.maxpool(conv6)
        conv7, conv8, conv9 = self.db3(mp2)

        mp3 = self.maxpool(conv9)
        conv10, conv11, conv12 = self.db4(mp3)

        mp4 = self.maxpool(conv12)
        conv13, conv14, conv15 = self.db5(mp4)

        uconv1 = self.ub1(conv15)
        uconv1 = torch.concatenate([uconv1, conv12], dim=1)
        conv16, conv17, conv18 = self.cb1(uconv1)

        uconv2 = self.ub2(conv18)
        uconv2 = torch.concatenate([uconv2, conv9], dim=1)
        conv19, conv20, conv21 = self.cb2(uconv2)

        uconv3 = self.ub3(conv21)
        uconv3 = torch.concatenate([uconv3, conv6], dim=1)
        conv22, conv23, conv24 = self.cb3(uconv3)

        uconv4 = self.ub4(conv24)
        uconv4 = torch.concatenate([uconv4, conv3], dim=1)
        conv25, conv26, conv27 = self.cb4(uconv4)

        out = self.ob1(conv27)

        return out
    

