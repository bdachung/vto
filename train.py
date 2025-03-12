import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from dataclasses import dataclass
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

@dataclass
class TrainingConfig:
    image_size = (1024, 768)  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-7
    lr_warmup_steps = 125
    save_image_epochs = 5
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "baselinev2"  # the model name locally and on the HF Hub
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_steps=2000
    num_samples=500

#     push_to_hub = True  # whether to upload the saved model to the HF Hub
#     hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
#     hub_private_repo = False
#     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

def total_trainable_params(model: torch.nn.Module):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def diffusion_loss(prediction, target):
    return torch.nn.functional.mse_loss(prediction, target)

def train_loop(config, unet, vae, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
        
    device = config.device

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    unet, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    
    print(f"VAE trainable params:", total_trainable_params(vae))
    print(f"Unet trainable params:", total_trainable_params(unet))

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            agnostic_image, clothing_image, person_image = batch
            # Sample noise to add to the images
            bs = person_image.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device,
                dtype=torch.int64
            )

            person_image = vae.encode(person_image).latent_dist.sample() 
            person_image *= vae.config.scaling_factor
            agnostic_image = vae.encode(agnostic_image).latent_dist.sample() 
            agnostic_image *= vae.config.scaling_factor
            clothing_image = vae.encode(clothing_image).latent_dist.sample()
            clothing_image *= vae.config.scaling_factor
            
            noise = torch.randn(person_image.shape, device=device)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_image = noise_scheduler.add_noise(person_image, noise, timesteps)
            
            inputs = torch.cat([noisy_image, agnostic_image, clothing_image], dim=1).to(device)
                
            with accelerator.accumulate(unet):
                # Predict the noise residual
                noise_pred = unet(inputs, timesteps).sample
                # noisy_dim = noisy_image.shape[-1]
                # noise_pred = noise_pred[..., :noisy_dim]
                loss = diffusion_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
            if global_step == config.num_steps:
                break
            
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            unet.save_pretrained(os.path.join(config.output_dir, "unet_stabilityai"))
            
        if global_step == config.num_steps:
            break

class VTONHDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, feature_extractor=None):
        """
        :param annotations_file: Đường dẫn đến file chứa các cặp ảnh
        :param img_dir: Thư mục chứa ảnh
        :param transform: Các phép biến đổi cho ảnh
        :param feature_extractor: Mô hình trích xuất đặc trưng cho trang phục
        """
        with open(annotations_file, 'r') as f:
            self.image_pairs = [line.strip().split() for line in f.readlines()][:config.num_samples]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        person_path, clothing_path = self.image_pairs[idx]
        clothing_image = Image.open(f"{self.img_dir}/cloth/{clothing_path}").convert("RGB")
        agnostic_image = Image.open(f"{self.img_dir}/agnostic-v3.2/{person_path}").convert("RGB")
        person_image = Image.open(f"{self.img_dir}/image/{person_path}").convert("RGB")

        if self.transform:
            agnostic_image = self.transform(agnostic_image)
            clothing_image = self.transform(clothing_image)
            person_image = self.transform(person_image)

        return agnostic_image, clothing_image, person_image

# Biến đổi cho ảnh và trích xuất đặc trưng trang phục
transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Tạo DataLoader
DATASET_DIR = "../datasets/viton-hd"
dataset = VTONHDDataset(annotations_file=os.path.join(".", 'VITONHD_train_paired.txt'), img_dir=os.path.join(DATASET_DIR, 'train'), 
                        transform=transform, feature_extractor=None)
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

unet = UNet2DModel(
    in_channels=12,
    out_channels=4,
    # Sử dụng block_out_channels để chỉ định số lượng kênh trong mỗi block
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",  # A downsampling layer with spatial resolution reduction
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # A downsampling layer with spatial resolution reduction and attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # An upsampling layer with spatial resolution increase
        "AttnUpBlock2D",  # An upsampling layer with spatial resolution increase and attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

unet = unet.from_pretrained("./baseline/unet_stabilityai")

# unet = nn.DataParallel(unet)

unet = unet.to(config.device)

vae = vae.to(config.device)

for param in vae.parameters():
    param.requires_grad = False

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

args = (config, unet, vae, noise_scheduler, optimizer, dataloader, lr_scheduler)

train_loop(*args)