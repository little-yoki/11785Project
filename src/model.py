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

# Fuse block 
class Fuse_conv(nn.Module):
    def __init__(self, in_rgb, in_depth, out_channels, dropout=False):
        super(Fuse_conv, self).__init__()
        self.encode_rgb = Double_conv(in_rgb, out_channels, dropout)
        self.encode_depth = Double_conv(in_depth, out_channels, dropout)
        self.concat_conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

        # down sampling
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, rgb_input, depth_input):
        rgb_output = self.encode_rgb(rgb_input)
        depth_output = self.encode_depth(depth_input)
        #total_output = rgb_output + depth_output
        total_output = torch.cat([rgb_output, depth_output], dim=1)
        total_output = self.concat_conv(total_output)
        total_output = self.bn(total_output)
        total_output = self.elu(total_output)
        rgb_output = self.down(total_output)
        depth_output = self.down(depth_output)
        return rgb_output, depth_output, total_output

class Fuse_UNet(nn.Module):
    def __init__(self, in_channels, classes, dropout=False, bilinear=False):
        super(Fuse_UNet, self).__init__()
        
        # encoding
        self.fuse_block1 = Fuse_conv(in_channels-1, 1, 64, dropout)
        self.fuse_block2 = Fuse_conv(64, 64, 128, dropout)
        self.fuse_block3 = Fuse_conv(128, 128, 256, dropout)
        self.fuse_block4 = Fuse_conv(256, 256, 512, dropout)
        self.fuse_block5 = Fuse_conv(512, 512, 1024, dropout)
        
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
        rgb_input = x[:, :-1]
        depth_input = x[:, -1:]
        rgb_out, depth_out, encode1 = self.fuse_block1(rgb_input, depth_input)
        rgb_out, depth_out, encode2 = self.fuse_block2(rgb_out, depth_out)
        rgb_out, depth_out, encode3 = self.fuse_block3(rgb_out, depth_out)
        rgb_out, depth_out, encode4 = self.fuse_block4(rgb_out, depth_out)
        rgb_out, depth_out, encode5 = self.fuse_block5(rgb_out, depth_out)
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

