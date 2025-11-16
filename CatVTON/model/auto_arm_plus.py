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
from model.target_cloth_masker import ClothMasker as TargetClothMasker
from model.skeleton_extractor import SkeletonExtractor
from model.DensePose import DensePose
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers.image_processor import VaeImageProcessor
import cv2

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask

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
    
current_dir = os.path.dirname(os.path.abspath(__file__))

class AutoARMPlus:
    def __init__(self, 
        model_ckp_path=os.path.join(current_dir, "MaskRefiner/epoch5"), 
        densepose_ckpt=os.path.join(current_dir, "AutoARMPlus/halfbody"),
        device="gpu"
    ):
        unet = AttentionUNet(in_channels=20, out_channels=1, num_channels=[128, 128, 256, 256, 512, 512])
        try:
            unet.load_state_dict(torch.load(model_ckp_path, weights_only=True, map_location="cpu"))
        except:
            unet = nn.DataParallel(unet)
            unet.load_state_dict(torch.load(model_ckp_path, weights_only=True, map_location="cpu"))
        unet.to(device)
        self.unet = unet
        self.skeleton_extractor = SkeletonExtractor()
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.device = device
        self.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def create_mask(self, image: Image.Image, cloth_type="upper_body"):
      """
      Return a gray image (mode=L)
      """
      inputs = self.seg_processor(images=image, return_tensors="pt").to(self.device)
      outputs = self.seg_model(**inputs)
      logits = outputs.logits.cpu()
      upsampled_logits = nn.functional.interpolate(
          logits,
          size=image.size[::-1],
          mode="bilinear",
          align_corners=False,
      )

      pred_seg = upsampled_logits.argmax(dim=1)[0]

      # Assuming class 4 is upper clothes
      if cloth_type == "lower_body":
        clothes_mask = ((pred_seg == 5) | (pred_seg == 6)).byte() * 255
      elif cloth_type == "dresses":
        clothes_mask = ((pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7)).byte() * 255
      else:
        clothes_mask = (pred_seg == 4).byte() * 255

      # Apply dilation to expand the mask
      kernel = np.ones((21, 21), dtype=np.uint8)
      clothes_mask_dilated = binary_dilation(clothes_mask.cpu().numpy(), structure=kernel).astype(np.uint8) * 255

      # Convert the binary mask to a PIL Image and save
      mask_image_clothes = Image.fromarray(clothes_mask_dilated, mode='L')

      # Ensure mask is the same size as the image and save
      mask_resized = mask_image_clothes.resize(image.size)

      return mask_resized

    def get_mask(self, image: Image.Image, cloth_type: str):
      """
      Return a gray image (mode=L)
      """
      if cloth_type not in ["upper_body", "lower_body", "dresses"]:
        raise ValueError("cloth type must be upper_body, lower_body or dresses")
      return self.create_mask(image, cloth_type)
    
    def get_skeleton(self, image: Image.Image):
      """
      Return an RGB image
      """
      skeleton_points, skeleton_image = self.skeleton_extractor(image)
      return Image.fromarray(np.uint8(skeleton_image)).convert("RGB")
    
    def get_densepose(self, image: Image.Image):
      return self.densepose_processor(image, resize=1024)

    def predict_from_original(
        self, 
        person_image: Union[str, Image.Image], 
        clothing_image: Union[str, Image.Image],
        cloth_type: str = "upper_body",
        image_size=(512, 384),
        is_full=True,
    ):
        if isinstance(person_image, str):
            person_image = Image.open(person_image)
        if isinstance(clothing_image, str):
            clothing_image = Image.open(clothing_image)
        
        person_mask_image = self.get_mask(person_image, cloth_type)
        person_pose_image = self.get_skeleton(person_image)
        person_dense_image = self.get_densepose(person_image)

        clothing_mask_image = self.get_mask(clothing_image, cloth_type)
        clothing_pose_image = self.get_skeleton(clothing_image)
        clothing_dense_image = self.get_densepose(clothing_image)
        
        result = self(
          person_image, 
          clothing_image, 
          person_mask_image,
          clothing_mask_image,
          person_pose_image,
          clothing_pose_image,
          person_dense_image,
          clothing_dense_image,
          cloth_type,
          image_size, 
          is_full
        )
        
        return {
          "person_image": person_image, 
          "clothing_image": clothing_image, 
          "person_mask_image": person_mask_image,
          "clothing_mask_image": clothing_mask_image,
          "person_pose_image": person_pose_image,
          "clothing_pose_image": clothing_pose_image,
          "person_dense_image": person_dense_image,
          "clothing_dense_image": clothing_dense_image,
          "result": result
        }
        
            
    def __call__(self, 
        person_image: Union[str, Image.Image],
        clothing_image: Union[str, Image.Image],
        person_mask_image: Union[str, Image.Image], 
        clothing_mask_image: Union[str, Image.Image], 
        person_pose_image: Union[str, Image.Image], 
        clothing_pose_image: Union[str, Image.Image],
        person_dense_image: Union[str, Image.Image],
        clothing_dense_image: Union[str, Image.Image],
        cloth_type: str = "upper_body", 
        image_size=(512, 384),
        is_full=True
    ) -> Image.Image:
        if isinstance(person_mask_image, str):
            person_mask_image = Image.open(person_mask_image)
        if isinstance(person_pose_image, str):
            person_pose_image = Image.open(person_pose_image)
        if isinstance(person_dense_image, str):
            person_dense_image = Image.open(person_dense_image)

        if isinstance(clothing_mask_image, str):
            clothing_mask_image = Image.open(clothing_mask_image)
        if isinstance(clothing_pose_image, str):
            clothing_pose_image = Image.open(clothing_pose_image)
        if isinstance(clothing_dense_image, str):
            clothing_dense_image = Image.open(clothing_dense_image)
        
        person_image = person_image.convert("RGB")
        person_mask_image = person_mask_image.convert("L")
        person_pose_image = person_pose_image.convert("RGB")
        person_dense_image = person_dense_image.convert("RGB")

        clothing_image = clothing_image.convert("RGB")
        clothing_mask_image = clothing_mask_image.convert("L")
        clothing_pose_image = clothing_pose_image.convert("RGB")
        clothing_dense_image = clothing_dense_image.convert("RGB")
        
        person_image = self.vae_processor.preprocess(person_image, image_size[0], image_size[1])[0]
        person_mask_image = self.mask_processor.preprocess(person_mask_image, image_size[0], image_size[1])[0]
        person_pose_image = self.vae_processor.preprocess(person_pose_image, image_size[0], image_size[1])[0]
        person_dense_image = self.vae_processor.preprocess(person_dense_image, image_size[0], image_size[1])[0]

        clothing_image = self.vae_processor.preprocess(clothing_image, image_size[0], image_size[1])[0]
        clothing_mask_image = self.mask_processor.preprocess(clothing_mask_image, image_size[0], image_size[1])[0]
        clothing_pose_image = self.vae_processor.preprocess(clothing_pose_image, image_size[0], image_size[1])[0]
        clothing_dense_image = self.vae_processor.preprocess(clothing_dense_image, image_size[0], image_size[1])[0]

        person_image = person_image.unsqueeze(0)
        person_mask_image = person_mask_image.unsqueeze(0)
        person_pose_image = person_pose_image.unsqueeze(0)
        person_dense_image = person_dense_image.unsqueeze(0)

        clothing_image = clothing_image.unsqueeze(0)
        clothing_mask_image = clothing_mask_image.unsqueeze(0)
        clothing_pose_image = clothing_pose_image.unsqueeze(0)
        clothing_dense_image = clothing_dense_image.unsqueeze(0)
        
        with torch.no_grad():
            input = torch.cat(
              [
                person_image, 
                person_mask_image, 
                person_pose_image,
                person_dense_image,
                clothing_image, 
                clothing_mask_image, 
                clothing_pose_image,
                clothing_dense_image,
              ], dim=1
            ).to(self.device)
            mask_pred = self.unet(input)
            
        mask_pred = mask_pred.squeeze(0).squeeze(0).to("cpu").numpy() > 0.7
    
        # if is_full:
        #     padding = 15
        # else:
        #     padding = 15

        h, w = image_size

        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        
        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1

        person_mask_image = np.array(person_mask_image).squeeze(0).squeeze(0).astype(np.uint8)
        print("Shape:", person_mask_image.shape)
        print("Max:", person_mask_image.max())
        person_mask_image = hull_mask(person_mask_image * 255) // 255  # Convex Hull to expand the mask area
        person_mask_image = cv2.GaussianBlur(person_mask_image * 255, (kernal_size, kernal_size), 0)
        person_mask_image[person_mask_image < 25] = 0
        person_mask_image[person_mask_image >= 25] = 1
        person_mask_image = cv2.dilate(person_mask_image, dilate_kernel, iterations=1)

        mask_pred = mask_pred.astype(np.uint8)
        mask_pred = hull_mask(mask_pred * 255) // 255  # Convex Hull to expand the mask area
        mask_pred = cv2.GaussianBlur(mask_pred * 255, (kernal_size, kernal_size), 0)
        mask_pred[mask_pred < 25] = 0
        mask_pred[mask_pred >= 25] = 1
        mask_pred = cv2.dilate(mask_pred, dilate_kernel, iterations=1)
      
        mask_pred = np.bitwise_or(mask_pred, person_mask_image).astype(np.uint8)

        mask_pred = binary_dilation(mask_pred, structure=dilate_kernel).astype(np.uint8)

        mask_pred_img = Image.fromarray(mask_pred).convert("RGB")
    
        return mask_pred_img
