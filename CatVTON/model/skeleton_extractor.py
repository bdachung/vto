import sys
sys.path.append('/home/jupyter/vinhdq_phucnph/OOTDiffusion/')

from preprocess.openpose.run_openpose import OpenPose  # Import the OpenPose class
from typing import Union
from PIL import Image

class SkeletonExtractor:
    def __init__(self):
        self.openpose_model = OpenPose(gpu_id=0)  # Assuming GPU 0 is available
    def __call__(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        keypoints, openpose_image = self.openpose_model(image)
        return keypoints, openpose_image