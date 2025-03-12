import sys
sys.path.append('/home/jupyter/vinhdq_phucnph/CatVTON/model/openpose/build/python')
sys.path.append('/home/jupyter/hungbd/image-background-remove-tool/')
sys.path.append('/home/jupyter/vinhdq_phucnph/OOTDiffusion/')
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from model.cloth_masker import AutoMasker

from preprocess.openpose.run_openpose import OpenPose  # Import the OpenPose class

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Agnostic Mask with DensePose, OpenPose, and CarveKit")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to generate mask.")
    parser.add_argument("--cloth_path", type=str, required=True, help="Path to the cloth to cloth mask.")
    parser.add_argument("--repo_path", type=str, default="zhengchong/CatVTON", help="Path or repo name of CatVTON.")
    parser.add_argument("--cloth_type", type=str, choices=['upper', 'lower', 'overall'], required=True, help="Clothing type.")
    parser.add_argument("--resize_width", type=int, default=384, help="Width to resize the image and mask.")
    parser.add_argument("--resize_height", type=int, default=512, help="Height to resize the image and mask.")
    return parser.parse_args()

def resize_image(image, width, height):
    """Resize the input image to the given width and height."""
    return image.resize((width, height), Image.Resampling.LANCZOS)

def initialize_carvekit():
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

def process_carvekit(cloth_path, interface, output_dir):
    """Process image with CarveKit and save mask."""
    image = Image.open(cloth_path)
    image = image.convert("RGB")

    # Generate mask
    mask = interface([image])[0]

    # Convert to grayscale (binary mask)
    mask_np = np.array(mask)[..., :3]
    idx = (mask_np[..., 0] == 130) & (mask_np[..., 1] == 130) & (mask_np[..., 2] == 130)
    mask_np = np.ones(idx.shape) * 255
    mask_np[idx] = 0
    mask_final = Image.fromarray(np.uint8(mask_np), 'L')

    # Save mask
    mask_output_path = os.path.join(output_dir, os.path.basename(cloth_path).replace('.jpg', '_carvekit_mask.png').replace('.jpeg', '_mask.png'))
    mask_final.save(mask_output_path)
    print(f"CarveKit mask saved to {mask_output_path}")

def main(args):
    # Download the repository
    args.repo_path = snapshot_download(repo_id=args.repo_path)

    # Initialize models
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(args.repo_path, "DensePose"),
        schp_ckpt=os.path.join(args.repo_path, "SCHP"),
        device='cuda',
    )
    carvekit_interface = initialize_carvekit()

    # Open and resize the input image
    input_image = Image.open(args.image_path)
    resized_image = resize_image(input_image, args.resize_width, args.resize_height)

    # Save resized image
    resized_image_path = args.image_path.replace('.jpg', '_resized.jpg').replace('.jpeg', '_resized.jpg')
    resized_image.save(resized_image_path)
    print(f"Resized image saved to {resized_image_path}")

    # Process DensePose and SCHP
    processed_results = automasker(resized_image, args.cloth_type)
    resized_mask = resize_image(processed_results['mask'], args.resize_width, args.resize_height)
    resized_densepose = resize_image(processed_results['densepose'], args.resize_width, args.resize_height)

    # Save DensePose and mask
    mask_output_path = args.image_path.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
    resized_mask.save(mask_output_path)
    print(f"Mask saved to {mask_output_path}")

    densepose_output_path = args.image_path.replace('.jpg', '_densepose.png').replace('.jpeg', '_densepose.png')
    resized_densepose.save(densepose_output_path)
    print(f"Densepose saved to {densepose_output_path}")

    # Process OpenPose using the integrated class
    openpose_model = OpenPose(gpu_id=0)  # Assuming GPU 0 is available
    keypoints, openpose_image = openpose_model(input_image)

    # Save OpenPose keypoints and skeleton visualization
    openpose_output_path = args.image_path.replace('.jpg', '_openpose.png').replace('.jpeg', '_openpose.png')
    cv2.imwrite(openpose_output_path, cv2.cvtColor(openpose_image, cv2.COLOR_RGB2BGR))
    print(f"OpenPose skeleton saved to {openpose_output_path}")

    # Process CarveKit
    output_dir = os.path.dirname(args.image_path)
    process_carvekit(args.cloth_path, carvekit_interface, output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
