import argparse
import os
from huggingface_hub import snapshot_download
from PIL import Image
from model.cloth_masker import AutoMasker


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of Preprocess Agnostic Mask")
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the image to generate mask."
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="zhengchong/CatVTON",
        help=( "The Path or repo name of CatVTON. "),
    )
    parser.add_argument(
        "--cloth_type",
        type=str,
        choices=['upper', 'lower', 'overall'],
        required=True,
        help="Specify the type of clothing ('upper', 'lower', or 'overall')"
    )
    parser.add_argument(
        "--resize_width", 
        type=int, 
        default=768, 
        help="Width to resize the image and mask (default: 256)"
    )
    parser.add_argument(
        "--resize_height", 
        type=int, 
        default=1024, 
        help="Height to resize the image and mask (default: 256)"
    )
    args = parser.parse_args()
    return args

def resize_image(image, width, height):
    """Resize the input image to the given width and height."""
    return image.resize((width, height), Image.Resampling.LANCZOS)

def main(args):
    # Download the repository
    args.repo_path = snapshot_download(repo_id=args.repo_path)

    # Initialize the AutoMasker with necessary checkpoints
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(args.repo_path, "DensePose"),
        schp_ckpt=os.path.join(args.repo_path, "SCHP"),
        device='cuda', 
    )

    # Open the input image and resize it
    input_image = Image.open(args.image_path)
    resized_image = resize_image(input_image, args.resize_width, args.resize_height)

    # Save resized image (optional, for inspection)
    resized_image_path = args.image_path.replace('.jpg', '_resized.jpg').replace('.jpeg', '_resized.jpg')
    resized_image.save(resized_image_path)
    print(f"Resized image saved to {resized_image_path}")

    # Process the resized image to generate the mask and densepose
    processed_results = automasker(resized_image, args.cloth_type)
    mask = processed_results['mask']
    densepose = processed_results['densepose']  # Assume 'densepose' is returned by the model

    # Resize the mask and densepose to the same dimensions as the input image
    resized_mask = resize_image(mask, args.resize_width, args.resize_height)
    resized_densepose = resize_image(densepose, args.resize_width, args.resize_height)

    # Save the generated mask
    mask_output_path = args.image_path.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
    resized_mask.save(mask_output_path)
    print(f"Mask saved to {mask_output_path}")

    # Save the generated densepose
    densepose_output_path = args.image_path.replace('.jpg', '_densepose.png').replace('.jpeg', '_densepose.png')
    resized_densepose.save(densepose_output_path)
    print(f"Densepose saved to {densepose_output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
