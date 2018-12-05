#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, classes, dropout=False, bilinear=False):
        super(UNet, self).__init__()
        
        # encoding
        self.encode_layer1 = Double_conv(in_channels, 64, dropout)
        self.encode_layer2 = Double_conv(64, 128, dropout)
        self.encode_layer3 = Double_conv(128, 256, dropout)
        self.encode_layer4 = Double_conv(256, 512, dropout)
        self.encode_layer5 = Double_conv(512, 1024, dropout)
        
        #decoding
        self.decode_layer1 = Concat_conv(512)
        self.decode_layer2 = Concat_conv(256)
        self.decode_layer3 = Concat_conv(128)
        self.decode_layer4 = Concat_conv(64)
        
        #up sampling
        self.up1 = Upsample(1024, 512, bilinear)
        self.up2 = Upsample(512, 256, bilinear)
        self.up3 = Upsample(256, 128, bilinear)
        self.up4 = Upsample(128, 64, bilinear)
        
        # down sampling
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #prediction
        self.output_conv = nn.Conv2d(64, classes, kernel_size=1, bias=False)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        encode1 = self.encode_layer1(x)
        out = self.down(encode1)
        encode2 = self.encode_layer2(out)
        out = self.down(encode2)
        encode3 = self.encode_layer3(out)
        out = self.down(encode3)
        encode4 = self.encode_layer4(out)
        out = self.down(encode4)
        encode5 = self.encode_layer5(out)
        decode = self.up1(encode5)
        decode = self.decode_layer1(encode4, decode)
        decode = self.up2(decode)
        decode = self.decode_layer2(encode3, decode)
        decode = self.up3(decode)
        decode = self.decode_layer3(encode2, decode)
        decode = self.up4(decode)
        decode = self.decode_layer4(encode1, decode)
        out = self.output_conv(decode)
        
        return out

class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Double_conv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(p=0.2)
        self.elu = nn.ELU(inplace=True)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.elu(out)
        if(self.dropout):
            out = self.drop(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Upsample, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        
    def forward(self, x):
        out = self.up(x)
        return out
    
class Concat_conv(nn.Module):
    def __init__(self, layer_size):
        super(Concat_conv, self).__init__()
        self.conv = Double_conv(layer_size*2, layer_size)
        
    def forward(self, encoder_layer, decoder_layer):
        out = torch.cat([encoder_layer, decoder_layer], dim=1)
        out = self.conv(out)
        return out

# test model 
#x = Variable(torch.FloatTensor(np.random.random((1, 3, 480, 640))))
#net = UNet(3, 20)
#out = net(x)
