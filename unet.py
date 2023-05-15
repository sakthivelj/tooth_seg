import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super().__init__()
                
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv3d(64, n_classes, 1)
        
        
    def forward(self, x):
        # Save the original size
        original_size = x.size()[2:]
        
        # Find the nearest size that's divisible by 16
        target_size = [int(np.ceil(d / 16) * 16) for d in original_size]
        
        # Pad the input tensor to the target size
        x = F.pad(x, [0, target_size[2] - original_size[2], 0, target_size[1] - original_size[1], 0, target_size[0] - original_size[0]])
        # print(f"Padded input size: {x.size()}")
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        # print(f"Down 1 size: {x.size()}")

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        # print(f"Down 2 size: {x.size()}")
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        # print(f"Down 3 size: {x.size()}")
        
        x = self.dconv_down4(x)
        # print(f"Down 4 size: {x.size()}")
        
        x = self.upsample(x)
        # print(f"Up 1 size: {x.size()}")  
        
        x = torch.cat([x, conv3], dim=1)
        # print(f"Concat 1 size: {x.size()}")  
        
        x = self.dconv_up3(x)
        x = self.upsample(x)  
        # print(f"Up 2 size: {x.size()}")  
        
        x = torch.cat([x, conv2], dim=1) 
        # print(f"Concat 2 size: {x.size()}")   

        x = self.dconv_up2(x)
        x = self.upsample(x) 
        # print(f"Up 3 size: {x.size()}")  
        
       
        x = torch.cat([x, conv1], dim=1)
        # print(f"Concat 3 size: {x.size()}")   
        
        x = self.dconv_up1(x)
        # print(f"Up 4 size: {x.size()}")  
        
        out = self.conv_last(x)
        # print(f"Output before crop size: {out.size()}")
        
        # Crop the output tensor to the original size
        out = out[:, :, :original_size[0], :original_size[1], :original_size[2]]
        # print(f"Final output size: {out.size()}")
        
        return out
