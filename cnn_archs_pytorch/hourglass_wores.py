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


class Hourglass_wores(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super().__init__()
        self.db1 = ConvBlock(in_channels, out_channels)
        self.db2 = ConvBlock(out_channels, 2*out_channels)
        self.db3 = ConvBlock(2*out_channels, 4*out_channels)
        self.db4 = ConvBlock(4*out_channels, 8*out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sb1 = ConvBlock(out_channels, 2*out_channels)
        self.sb2 = ConvBlock(2*out_channels, 4*out_channels)
        self.sb3 = ConvBlock(4*out_channels, 8*out_channels)

        self.ub1 = ConvBlockTranspose(8*out_channels, 8*out_channels)
        self.cb1 = ConvBlock(8*out_channels, 4*out_channels)
        self.ub2 = ConvBlockTranspose(4*out_channels, 4*out_channels)
        self.cb2 = ConvBlock(4*out_channels, 2*out_channels)
        self.ub3 = ConvBlockTranspose(2*out_channels, 2*out_channels)
        self.cb3 = ConvBlock(2*out_channels, out_channels)

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

        conv13, conv14, conv15 = self.sb1(conv3)
        conv16, conv17, conv18 = self.sb2(conv6)
        conv19, conv20, conv21 = self.sb3(conv9)
        

        mp4 = self.maxpool(conv21)
        conv12 = conv12 + mp4
        conv22 = self.ub1(conv12)
        conv23, conv24, conv25 = self.cb1(conv22)

        mp5 = self.maxpool(conv18)
        conv25 = conv25 + mp5
        conv26 = self.ub2(conv25)
        conv27, conv28, conv29 = self.cb2(conv26)

        mp6 = self.maxpool(conv15)
        conv29 = conv29 + mp6
        conv30 = self.ub3(conv29)
        conv31, conv32, conv33 = self.cb3(conv30)

        out = self.ob1(conv33)

        return out



