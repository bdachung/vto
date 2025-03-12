import sys
sys.path.append('../vinhdq_phucnph/CatVTON')
sys.path.append('../vinhdq_phucnph/CatVTON/model')
from model.cloth_masker import AutoMasker
from model.mask_refiner import MaskRefiner

import argparse
import os
from datetime import datetime
import time

import gradio as gr
from PIL import ImageFilter
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

from model.cloth_masker import AutoMasker, vis_mask
from model.mask_refiner import MaskRefiner
from model.skeleton_extractor import SkeletonExtractor
from model.target_cloth_masker import ClothMasker as TargetClothMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def zoom_out_image(image, scale=0.75, background_color="white"):
    """
    Zoom out an image by scaling it down and adding padding.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the zoomed-out image.
    :param scale: Scale factor for zooming out (e.g., 0.8 for 80% of original size).
    :param background_color: Background color for padding.
    """

    original_size = image.size  # (width, height)

    # Calculate new dimensions for the image
    new_width = int(original_size[0] * scale)
    new_height = int(original_size[1] * scale)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a padded image with the original dimensions
    padded_image = Image.new("RGB", original_size, background_color)
    paste_position = (
        (original_size[0] - new_width) // 2,
        (original_size[1] - new_height) // 2
    )
    padded_image.paste(resized_image, paste_position)

    return padded_image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def repaint(person, mask, result):
    _, h = result.size
    # kernal_size = h // 50
    # if kernal_size % 2 == 0:
    #     kernal_size += 1
    # mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

maskrefiner_half = MaskRefiner(model_ckp_path="/home/jupyter/hungbd/mask_checkpoint/version4/mask3", device="cuda")
maskrefiner_full = MaskRefiner(device="cuda")
skeleton_extractor = SkeletonExtractor()
target_cloth_masker = TargetClothMasker()

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5

def need_zoom_out(cloth_image, th=0.6):
    cloth_mask = target_cloth_masker(cloth_image)
    tmp = np.array(cloth_mask)
    tmp = tmp / tmp.max()
    percent = tmp.sum() / tmp.size
    # print(percent > 0.6)
    return percent > th

def check_full(person_image):
    ske_points, _ = skeleton_extractor(person_image)
    keypoints = ske_points['pose_keypoints_2d']
    # Define keypoints
    hip_left = keypoints[8]
    hip_right = keypoints[11]
    knee_left = keypoints[10]
    knee_right = keypoints[13]
    ankle_left = keypoints[9]
    ankle_right = keypoints[12]
    
    # Check if points are valid (not [0, 0])
    if all([hip_left[1] > 0, hip_right[1] > 0, knee_left[1] > 0, knee_right[1] > 0]):
        # Calculate distances
        hip_to_knee_left = calculate_distance(hip_left, knee_left)
        hip_to_knee_right = calculate_distance(hip_right, knee_right)
        knee_to_end_left = 0 if ankle_left[1] == 0 else calculate_distance(knee_left, ankle_left)
        knee_to_end_right = 0 if ankle_right[1] == 0 else calculate_distance(knee_right, ankle_right)
        
        # Compare distances
        if knee_to_end_left == 0 and knee_to_end_right == 0:
            return False  # Missing ankles, likely half body
        elif abs(knee_to_end_left - hip_to_knee_left) < 0.2 * hip_to_knee_left and \
             abs(knee_to_end_right - hip_to_knee_right) < 0.2 * hip_to_knee_right:
            return False  # Ratios match for half body
        else:
            return True
    return False

def predict(person_image, cloth_image, category, image_size=(1024, 512), device="cuda"):   
    h, w = image_size
    original_size = person_image.size

    assert category in ["upper", "lower", "overall"]
    
    if category != "overall":
        agnostic_mask_image = automasker(person_image, mask_type=category)['mask']
        if check_full(person_image):
            print("full")
            cloth_image = zoom_out_image(cloth_image, 0.75) if need_zoom_out(cloth_image, 0.6) else resize_and_padding(cloth_image, (args.width, args.height)) 
            pred_mask_image =  maskrefiner_full.predict_from_original(person_image, cloth_image, agnostic_mask_image, image_size=(512, 384), is_full=True)['result']
            final_result = automasker.create_refined_mask(person_image, pred_mask_image, mask_type=category)['result']
            mask = np.array(final_result.convert("L").resize(original_size))
        else:
            print("half")
            cloth_image = zoom_out_image(cloth_image, 0.9) if need_zoom_out(cloth_image, 0.75) else resize_and_padding(cloth_image, (args.width, args.height)) 
            pred_mask_image =  maskrefiner_half.predict_from_original(person_image, cloth_image, agnostic_mask_image, image_size=(512, 384), is_full=False)['result']
            final_result = automasker.create_refined_mask(person_image, pred_mask_image, mask_type=category)['result']
            mask = np.array(final_result.convert("L").resize(original_size))
        # pred_mask_image =  maskrefiner_half.predict_from_original(person_image, cloth_image, agnostic_mask_image, image_size=(512, 384), is_full=False)['result']
        # final_result = automasker.create_refined_mask(person_image, pred_mask_image, mask_type=category)['result']
        # mask = np.array(final_result.convert("L").resize(original_size))
    else:
        agnostic_mask_image = automasker(person_image, mask_type=category)['mask']
        mask = agnostic_mask_image.resize(original_size)
        mask = np.array(mask)
        
    # person_image = person_image.resize((w, h))
    person_image = np.array(person_image)
    person_image[mask > 0] = [128, 128, 128]
    person_image = Image.fromarray(person_image)

    return mask, person_image

def submit_function(person_image, cloth_image, cloth_type, num_inference_steps, progress=gr.update()):
    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = torch.Generator(device='cuda').manual_seed(555)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    progress = gr.update(value=10, visible=True)  # Hiển thị thanh tiến độ
    time.sleep(0.5)

    mask, final_mask_image = predict(person_image, cloth_image, cloth_type, (args.height, args.width))
    mask[mask > 0] = 255
    mask = Image.fromarray(mask).resize(person_image.size)

    progress = gr.update(value=50)
    time.sleep(0.5)

    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=2.5,
        generator=generator
    )[0]
    
    if args.repaint:
        result_image = repaint(person_image, mask, result_image)        

    progress = gr.update(value=80)
    time.sleep(0.5)

    final_mask_image = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, final_mask_image, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)

    progress = gr.update(value=100)  # Hoàn thành
    time.sleep(0.3)

    return result_image, progress, final_mask_image
    
def person_example_fn(image_path):
    return image_path

def list_valid_files(directory):
    """Lấy danh sách các tệp hợp lệ trong thư mục, bỏ qua thư mục con."""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))  # Chỉ lấy file, bỏ thư mục
        and not file.startswith('.')  # Bỏ qua các file ẩn (VD: .ipynb_checkpoints)
        and file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))  # Chỉ lấy file ảnh
    ]


def app_gradio():
    with gr.Blocks(title="Auto ARM") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Row():
                    person_image = gr.Image(
                        interactive=True, label="Person Image", type="filepath", width=384, height=512
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=230):
                        cloth_image = gr.Image(
                            interactive=True, label="Condition Image", type="filepath"
                        )
                    with gr.Column(scale=1, min_width=120):
                        gr.Markdown(
                        )
                        cloth_type = gr.Radio(
                            label="Try-On Cloth Type",
                            choices=["upper", "overall"],
                            value="upper",
                        )


                submit = gr.Button("Submit")
                gr.Markdown(
                    '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
                )
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(
                        label="Inference Step", minimum=10, maximum=100, step=5, value=50
                    )
            with gr.Column(scale=1, min_width=250):
                final_mask_image = gr.Image(interactive=False, label="Mask")
                progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Processing...", visible=False)
                with gr.Row():
                    with gr.Column():
                        root_path = "resource/demo/example"
                        condition_upper_exm = gr.Examples(
                            examples=list_valid_files(os.path.join(root_path, "condition", "upper_female")),
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Upper Examples",
                        )
                        condition_person_exm = gr.Examples(
                            examples=list_valid_files(os.path.join(root_path, "condition", "overal_female")),
                            examples_per_page=4,
                            inputs=cloth_image,
                            label="Condition Overall Examples",
                        )
            with gr.Column(scale=1, min_width=250):
                result_image = gr.Image(interactive=False, label="Result")
                progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Processing...", visible=False)
                with gr.Row():
                    with gr.Column():
                        condition_person_exm = gr.Examples(
                            examples=list_valid_files(os.path.join(root_path, "condition", "person")),
                            examples_per_page=4,
                            inputs=person_image,
                            label="Person Examples",
                        )

            submit.click(
                submit_function,
                [person_image, cloth_image, cloth_type, num_inference_steps],
                [result_image, progress_bar, final_mask_image],
            )
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
