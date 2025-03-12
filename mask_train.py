import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import json

NUM_TRAIN_SAMPLES = 11647

@dataclass
class TrainingConfig:
    image_size = (512, 384)  # the generated image resolution
    train_batch_size = 6
    eval_batch_size = 6  # how many images to sample during evaluation
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 1
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "mask_checkpoint/version4"  # the model name locally and on the HF Hub
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
    num_steps=None
    num_samples=int(0.9*NUM_TRAIN_SAMPLES)

#     push_to_hub = True  # whether to upload the saved model to the HF Hub
#     hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
#     hub_private_repo = False
#     overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class FlexibleNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        num_channels = tensor.shape[-1]  # Get the number of channels
        adjusted_mean = self.mean[:num_channels]
        adjusted_std = self.std[:num_channels]
        return transforms.functional.normalize(tensor, mean=adjusted_mean, std=adjusted_std)

class VTONHDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, sub_dir, sample_range=None, transform=None):
        """
        :param annotations_file: Đường dẫn đến file chứa các cặp ảnh
        :param img_dir: Thư mục chứa ảnh
        :param transform: Các phép biến đổi cho ảnh
        :param feature_extractor: Mô hình trích xuất đặc trưng cho trang phục
        """
        print(sample_range)
        with open(annotations_file, 'r') as f:
            self.image_pairs = [line.strip().split() for line in f.readlines()][sample_range]
        self.img_dir = img_dir
        self.sub_dir = sub_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        person_path, clothing_path = self.image_pairs[idx]
        
        #target
        parse_v3_mask_image = Image.open(f"{self.sub_dir}/image_parse_v3_binary2/{clothing_path}").convert("L")
        # parse_v3_mask_image = np.expand_dims(parse_v3_mask_image, axis=-1)
        #input
        clothing_mask_image = Image.open(f"{self.img_dir}/cloth-mask/{clothing_path}").convert("RGB")
        # clothing_mask_image = np.expand_dims(clothing_mask_image, axis=-1)
        
        agnostic_mask_image = Image.open(f"{self.sub_dir}/agnostic_mask_binary/{clothing_path}").convert("RGB")
        # agnostic_mask_image = np.expand_dims(agnostic_mask_image, axis=-1)
        
        pose_image = Image.open(f"{self.img_dir}/openpose_img/{clothing_path.replace('.jpg', '_rendered.png')}").convert("RGB")
        
        if self.transform:
            # parse_v3_mask_image = self.transform(parse_v3_mask_image)
            clothing_mask_image = self.transform(clothing_mask_image)
            agnostic_mask_image = self.transform(agnostic_mask_image)
            pose_image = self.transform(pose_image)
            
        transform = transforms.Grayscale(num_output_channels=1)
            
        parse_v3_mask_image = transforms.ToTensor()(transforms.Resize(config.image_size)(parse_v3_mask_image))
        clothing_mask_image = transform(clothing_mask_image)
        agnostic_mask_image = transform(agnostic_mask_image)

        return parse_v3_mask_image, clothing_mask_image, agnostic_mask_image, pose_image

# Biến đổi cho ảnh và trích xuất đặc trưng trang phục
transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Tạo DataLoader
DATASET_DIR = "../datasets/viton-hd"
SUB_DIR = "./mask_train_dataset"
dataset = VTONHDDataset(annotations_file=os.path.join(".", 'VITONHD_train_paired.txt'), 
                        img_dir=os.path.join(DATASET_DIR, 'train'), 
                        sub_dir=SUB_DIR,
                        sample_range=slice(config.num_samples),
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

dataset = VTONHDDataset(annotations_file=os.path.join(".", 'VITONHD_train_paired.txt'), 
                        img_dir=os.path.join(DATASET_DIR, 'train'), 
                        sub_dir=SUB_DIR,
                        sample_range=slice(config.num_samples, NUM_TRAIN_SAMPLES),
                        transform=transform)
val_dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)

def iou_score(pred, target):
    """
    Calculate IOU
    Params:
    -------
        pred: torch tensor has shape (B, 1, H, W)
        target: torch tensor has shape (B, 1, H, W)
    Returns:
    --------
        iou (float)
    """
    area_pred = pred.sum(dim=[1, 2, 3])
    area_target = target.sum(dim=[1, 2, 3])
    intersection = pred * target
    area_intersection = intersection.sum(dim=[1, 2, 3])
    area_union = area_pred + area_target - area_intersection
    iou = area_intersection / area_union
    return iou.mean().cpu().item()

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

# Hàm mất mát
def mask_loss(prediction, target):
    return torch.nn.functional.binary_cross_entropy(prediction, target)

def train_loop(config, unet, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    
    # Initialize accelerator and tensorboard logging
    train_loss = []
    val_loss = []
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
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    
    print(f"Unet trainable params:", count_trainable_params(unet))

    # Now you train the model
    for epoch in range(config.num_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            parse_v3_mask_image, clothing_mask_image, agnostic_mask_image, pose_image = batch
            
            inputs = torch.cat([clothing_mask_image, agnostic_mask_image, pose_image], dim=1).to(device)
            
            # print(f"inputs shape:", inputs.shape)
                
            with accelerator.accumulate(unet):
                mask_pred = unet(inputs)
                loss = mask_loss(mask_pred.float(), parse_v3_mask_image.float())
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            train_loss.append(loss.detach().item())
            # print(f"step: {global_step}, loss: {loss.detach().item()}")
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            val = []
            for step, batch in enumerate(val_dataloader):
                parse_v3_mask_image, clothing_mask_image, agnostic_mask_image, pose_image = batch
                
                with torch.no_grad():
                    inputs = torch.cat([clothing_mask_image, agnostic_mask_image, pose_image], dim=1).to(device)
                    mask_pred = unet(inputs)
                    iou = iou_score(mask_pred > 0.5, parse_v3_mask_image)
                    val.append(iou)
            __val_loss = np.mean(val)
            val_loss.append(__val_loss)
            progress_bar.set_postfix({"Val_iou": __val_loss})
            
            early_stopping(-__val_loss, unet)
            
            if early_stopping.early_stop:
                print("early stopping")
                break
    
    torch.save(unet.state_dict(), os.path.join(config.output_dir, "mask3"))
    print("save model state dict")
    
    with open(os.path.join(config.output_dir, "train_loss.json"), 'w') as f:
        json.dump(train_loss, f, indent=4) 
    print("save train loss")
    
    with open(os.path.join(config.output_dir, "val_iou.json"), 'w') as f:
        json.dump(val_loss, f, indent=4) 
    print("save val IOU")
    
            
from diffusers.optimization import get_cosine_schedule_with_warmup

unet = AttentionUNet(in_channels=5, out_channels=1, num_channels=[128, 128, 256, 256, 512, 512])

unet.load_state_dict(torch.load("mask_checkpoint/mask3", weights_only=True))

optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

args = (config, unet, optimizer, dataloader, val_dataloader, lr_scheduler)

train_loop(*args)