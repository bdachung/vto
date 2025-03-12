import sys
sys.path.append('/home/jupyter/hungbd/image-background-remove-tool/')

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from typing import Union

class ClothMasker:
    def __init__(self):
        self.interface = self.initialize_carvekit()
    
    def __call__(self, cloth_path: Union[str, Image.Image]) -> Image.Image:
        """Process image with CarveKit and save mask."""
        if isinstance(cloth_path, str):
            image = Image.open(cloth_path)
            image = image.convert("RGB")
        else:
            image = cloth_path

        # Generate mask
        mask = self.interface([image])[0]

        # Convert to grayscale (binary mask)
        mask_np = np.array(mask)[..., :3]
        idx = (mask_np[..., 0] == 130) & (mask_np[..., 1] == 130) & (mask_np[..., 2] == 130)
        mask_np = np.ones(idx.shape) * 255
        mask_np[idx] = 0
        mask_final = Image.fromarray(np.uint8(mask_np), 'L')
        
        return mask_final
    
    def  initialize_carvekit(self):
        """Initialize CarveKit model."""
        config = MLConfig(
            segmentation_network="tracer_b7",
            preprocessing_method="none",
            postprocessing_method="fba",
            seg_mask_size=640,
            trimap_dilation=30,
            trimap_erosion=5,
            device='cuda'
        )
        return init_interface(config)