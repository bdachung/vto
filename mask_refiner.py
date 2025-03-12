import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_dilation
import json
from torchvision import transforms, models
from dataclasses import dataclass
from typing import Union, List, Dict

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels):
        super(AttentionUNet, self).__init__()
        assert len(num_channels) > 1, "num_channels must have at least 2 elements."

        self.num_channels = num_channels
        self.encoders = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Create encoders
        self.encoders.append(self.conv_block(in_channels, num_channels[0]))
        for i in range(1, len(num_channels) - 1):
            self.encoders.append(self.conv_block(num_channels[i - 1], num_channels[i]))

        # Center block (bottleneck)
        self.center = self.conv_block(num_channels[-2], num_channels[-1])

        # Create attention blocks and decoders
        for i in range(len(num_channels) - 2, -1, -1):
            self.attention_blocks.append(AttentionBlock(F_g=num_channels[i + 1], F_l=num_channels[i], F_int=num_channels[i] // 2))
            self.decoders.append(self.up_conv(num_channels[i + 1] + num_channels[i], num_channels[i]))

        # Final output layer
        self.final = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)
        
        # activate sigmoid
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        enc_features = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            x = F.max_pool2d(x, 2)
            enc_features.append(x)
            # print("downsample:", x.shape)

        # Center
        x = self.center(x)
        # print("center:", x.shape)

        # Decoder path
        for i in range(len(self.decoders)):
            # print("x", x.shape)
            # print("enc", enc_features[-(i + 1)].shape)
            x = self.decoders[i](torch.cat((self.attention_blocks[i](g=x, x=enc_features[-(i + 1)]), x), dim=1))

        # Final output
        x = self.final(x)
        
        x = self.sigmoid(x)
        
        return x
    
class MaskRefiner:
    def __init__(self, model_ckp_path):
        unet = AttentionUNet(in_channels=5, out_channels=1, num_channels=[128, 128, 256, 256, 512, 512])
        try:
            unet.load_state_dict(torch.load(model_ckp_path, weights_only=True, map_location="cpu"))
        except:
            unet = nn.DataParallel(unet)
            unet.load_state_dict(torch.load(model_ckp_path, weights_only=True, map_location="cpu"))
            
    def __call__(self, clothing_mask_image: Union[str, Image.Image], 
                       agnostic_mask_image: Union[str, Image.Image], 
                       pose_image: Union[str, Image.Image], 
                       image_size=(512, 384), 
                       device="cpu") -> Image.Image:
        if isinstance(clothing_mask_image, str):
            clothing_mask_image = Image.open(clothing_mask_image)
        if isinstance(agnostic_mask_image, str):
            agnostic_mask_image = Image.open(agnostic_mask_image)
        if isinstance(pose_image, str):
            pose_image = Image.open(pose_image)
        
        clothing_mask_image = clothing_mask_image.convert("RGB")
        agnostic_mask_image = agnostic_mask_image.convert("RGB")
        pose_image = pose_image.convert("RGB")
        
        transform_1 = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        transform2 = transforms.Grayscale(num_output_channels=1)
    
        clothing_mask_image = transform2(transform_1(clothing_mask_image))
        agnostic_mask_image = transform2(transform_1(agnostic_mask_image))
        pose_image = transform_1(pose_image)

        clothing_mask_image = clothing_mask_image.unsqueeze(0)
        agnostic_mask_image = agnostic_mask_image.unsqueeze(0)
        pose_image = pose_image.unsqueeze(0)
        
        with torch.no_grad():
            input = torch.cat([clothing_mask_image, agnostic_mask_image, pose_image], dim=1).to(config.device)
            mask_pred = unet(input)
            
        mask_pred = mask_pred.squeeze(0).squeeze(0).to("cpu").numpy() > 0.5
    
        # padding = 10
        # mask_pred = binary_dilation(mask_pred, structure=np.ones((padding, padding)))
    
        mask_pred_img = Image.fromarray(mask_pred).convert("RGB")
    
        return mask_pred_img
        
# def refine_mask(clothing_mask_image, agnostic_mask_image, pose_image):

#     cloth_mask_image = clothing_mask_image.convert("RGB")
#     agnostic_mask_image = agnostic_mask_image.convert("RGB")
#     pose_image = pose_image.convert("RGB")
    
#     transform_1 = transforms.Compose([
#         transforms.Resize(config.image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
        
#     transform2 = transforms.Grayscale(num_output_channels=1)
    
#     clothing_mask_image = transform2(transform_1(clothing_mask_image))
#     agnostic_mask_image = transform2(transform_1(agnostic_mask_image))
#     pose_image = transform_1(pose_image)
    
#     clothing_mask_image = clothing_mask_image.unsqueeze(0)
#     agnostic_mask_image = agnostic_mask_image.unsqueeze(0)
#     pose_image = pose_image.unsqueeze(0)
    
#     with torch.no_grad():
#         input = torch.cat([clothing_mask_image, agnostic_mask_image, pose_image], dim=1).to(config.device)
#         mask_pred = unet(input)
        
#     mask_pred = mask_pred.squeeze(0).squeeze(0).to("cpu").numpy() > 0.5
    
#     # padding = 10
#     # mask_pred = binary_dilation(mask_pred, structure=np.ones((padding, padding)))
    
#     mask_pred_img = Image.fromarray(mask_pred).convert("RGB")
    
#     return mask_pred_img