import os
import copy
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from models.denoiser import Denoiser
from utils import update_ema
from dataset_loader import load_dataset
from config import *
import wandb
import math
from inference import DiffusionTransformer
from pytorch_fid import fid_score
from torchvision.utils import save_image

def calculate_fid_for_generations(diffusion_transformer, batch_images, epoch, save_img_dir):
    """
    Calculate FID score between batch images and generated images
    Args:
        diffusion_transformer: The diffusion model
        batch_images: Tensor of shape [B, C, H, W] containing real images from the batch
        epoch: Current training epoch
        save_img_dir: Directory to save images
    """
    # Create directories for FID calculation
    latest_checkpoint = sorted(
        [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pt")],
        key=os.path.getmtime
    )[-1]

    checkpoint = torch.load(latest_checkpoint, map_location=diffusion_transformer.device)
    diffusion_transformer.model.load_state_dict(checkpoint.get("model_ema", checkpoint))
    print(f"Loaded model from {latest_checkpoint} for FID calculation at epoch {epoch}")

    # Proceed with FID calculation
    real_images_dir = os.path.join(save_img_dir, 'real_images')
    generated_images_dir = os.path.join(save_img_dir, f'generated_images_epoch_{epoch}')
    os.makedirs(real_images_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    # Clear previous images
    for dir_path in [real_images_dir, generated_images_dir]:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}: {e}')

    # Save 10 real images from the batch
    for idx in range(min(10, len(batch_images))):
        try:
            save_image(batch_images[idx], os.path.join(real_images_dir, f'real_img_{idx}.png'))
        except Exception as e:
            print(f"Failed to save real image {idx}: {e}")
            continue

    # Generate and save 10 synthetic images
    prompts = [
        "a red car on a black road looking majestic",
        "a blue sports car racing through city streets",
        "a vintage car parked near a sunset",
        "a white SUV driving through mountains",
        "a yellow convertible on a coastal road",
        "a black luxury sedan in front of a mansion",
        "a green jeep crossing a river",
        "a silver sports car on a race track",
        "a brown classic car in autumn scenery",
        "a purple supercar with city lights background"
    ]

    for idx, prompt in enumerate(prompts):
        try:
            img = diffusion_transformer.generate_image_from_text(
                prompt=prompt,
                class_guidance=6,
                seed=16 + idx,
                num_imgs=1,
                img_size=32  # Using default size from your DiffusionTransformer
            )
            img.save(os.path.join(generated_images_dir, f'gen_img_{idx}.png'))
        except Exception as e:
            print(f"Failed to generate image {idx}: {e}")
            continue

    # Calculate FID score
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_dir, generated_images_dir],
            batch_size=1,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        print(f"FID Score at epoch {epoch}: {fid_value:.2f}")
        return fid_value
    except Exception as e:
        print(f"Failed to calculate FID score: {e}")
        return None

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weighted_mse_loss(pred, target, noise_level):
    # Weight loss more heavily for less noisy samples
    weights = 1 / (noise_level.view(-1, 1, 1, 1) + 0.1)
    mse = F.mse_loss(pred, target, reduction='none')
    return (mse * weights).mean()


def get_noise_schedule(t, start=0, end=1):
    """
    Cosine noise schedule as proposed in Improved DDPM paper
    """
    # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    s = 0.008  # Offset to prevent singularity
    t = torch.as_tensor(t)
    return torch.cos(((t / end + s) / (1 + s)) * math.pi * 0.5) ** 2

def train(train_loader, val_loader=None):  # Add an optional validation loader
    # Check if CUDA is available
    if torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()  # Use default settings for CPU/MPS
    device = accelerator.device  # Use accelerator's device

    # Initialize WandB
    if train_use_wandb:
        wandb.init(
            project=wandb_project_name,
            config={
                "image_size": denoiser_image_size,
                "noise_embed_dims": denoiser_noise_embed_dims,
                "patch_size": denoiser_patch_size,
                "embed_dim": denoiser_embed_dim,
                "dropout": denoiser_dropout,
                "n_layers": denoiser_n_layers,
                "batch_size": train_batch_size,
                "learning_rate": train_lr,
                "epochs": train_n_epoch,
            },
        )

    # Define the Denoiser model
    model = Denoiser(
        image_size=denoiser_image_size,
        noise_embed_dims=denoiser_noise_embed_dims,
        patch_size=denoiser_patch_size,
        embed_dim=denoiser_embed_dim,
        dropout=denoiser_dropout,
        n_layers=denoiser_n_layers,
    ).to(device)
    total_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,  # Slightly higher initial learning rate
    weight_decay=0.03,  # Moderate weight decay
    betas=(0.9, 0.99)  # Modified beta2 for better adaptation
    )
    
    # More aggressive learning rate schedule
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=8e-4,  # Peak learning rate
        epochs=train_n_epoch,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,  # Longer warmup
        div_factor=8,  # Less aggressive initial lr reduction
        final_div_factor=75,
        anneal_strategy='cos'
    )

    # Restore checkpoint locally
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        key=os.path.getmtime,
    )
    global_step = 0
    if checkpoint_files:
        # Load the most recent checkpoint
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint, weights_only=True)
        model.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        print(f"Resumed training from step {global_step} using {latest_checkpoint}.")
    else:
        print("No checkpoint found locally. Starting training from scratch.")

    # Initialize EMA model
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)  # EMA model should not require gradients

    # Prepare data and models for Accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    # Initialize WandB tracking
    if train_use_wandb:
        accelerator.init_trackers(project_name=wandb_project_name, config=wandb.config)
    try:
        diffusion_transformer = DiffusionTransformer()
    except Exception as e:
        diffusion_transformer = None
        accelerator.print(f"Failed to initialize DiffusionTransformer: {e}")  # Initialize the inference model
    save_img_dir = "training_progress"
    os.makedirs(save_img_dir, exist_ok=True)
    # Training loop
    for epoch in range(1, train_n_epoch):
        accelerator.print(f"Epoch {epoch}/{train_n_epoch}")
        model.train()

        epoch_loss = 0
        batch_count = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # Move data to the correct device
            x, y = x.to(device), y.to(device)

            # Apply VAE scale factor if needed
            x = x.float().mul_(0.18215)

            batch_size = x.shape[0]
            
            # Sample random points in [0, 1] for the noise schedule
            t = torch.rand(batch_size, device=device)
            
            # Get noise levels from cosine schedule
            noise_level = get_noise_schedule(t)
            signal_level = torch.sqrt(1 - noise_level)  # Changed to sqrt for proper scaling
            
            # Generate noise and mix with signal
            noise = torch.randn_like(x, device=device)
            x_noisy = (noise_level.view(-1, 1, 1, 1) * noise + 
                      signal_level.view(-1, 1, 1, 1) * x)
            
            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            

            # Forward pass
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                pred = model(x_noisy, noise_level.view(-1, 1), y)
                loss = weighted_mse_loss(pred, x, noise_level)



                # Backward pass
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                optimizer.step()

                # Update EMA model
                update_ema(ema_model, model, alpha=0.999)

            # Accumulate loss for average computation
            epoch_loss += loss.item()
            batch_count += 1

            # Log and print loss
            if global_step % 10 == 0:  # Log every 10 steps
                accelerator.print(f"Step {global_step}: Loss = {loss.item():.6f}")
                if train_use_wandb:
                    wandb.log({"train_loss": loss.item(), "step": global_step})

            # Save and evaluate periodically
            if global_step % train_save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # Generate a unique checkpoint filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}_step{global_step}.pt")
                    
                    # Save the checkpoint
                    checkpoint = {
                        "model_ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                    }
                    torch.save(checkpoint, checkpoint_filename)

                    # Remove older checkpoints to keep only the latest 5
                    checkpoint_files.append(checkpoint_filename)
                    checkpoint_files = sorted(checkpoint_files, key=os.path.getmtime)
                    if len(checkpoint_files) > 5:
                        oldest_checkpoint = checkpoint_files.pop(0)
                        os.remove(oldest_checkpoint)

            global_step += 1

        # Calculate average loss
        avg_loss = epoch_loss / batch_count
        accelerator.print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
        if train_use_wandb:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        try:
            scheduler.step()
        except KeyError as e:
            if 'max_lr' in str(e):
                print("max_lr not found. Reinitializing OneCycleLR with warmup.")
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=1e-4,
                    epochs=train_n_epoch - epoch + 1,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.1  # Smaller warmup percentage to prevent too much warmup
                )
                scheduler.step()

   

        # Evaluate on validation set
        if val_loader:
            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    x = x.float().mul_(0.18215)
                    x = x.float()
                    batch_size = x.shape[0]
                    t = torch.rand(batch_size, device=device)
                    noise_level = get_noise_schedule(t)
                    signal_level = torch.sqrt(1 - noise_level)
                    
                    noise = torch.randn_like(x, device=device)
                    x_noisy = (noise_level.view(-1, 1, 1, 1) * noise + 
                            signal_level.view(-1, 1, 1, 1) * x)
                    
                    x_noisy = x_noisy.float()
                    noise_level = noise_level.float()

                    pred = model(x_noisy, noise_level.view(-1, 1), y)
                    loss = F.mse_loss(pred, x)
                    val_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = val_loss / val_batch_count
            accelerator.print(f"Epoch {epoch} validation loss: {avg_val_loss:.6f}")
            if train_use_wandb:
                wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
    if epoch % 3 == 0 and diffusion_transformer is not None:  # Generate and save images every 3 epochs
        try:
            # Use the current batch x that we already have
            fid_value = calculate_fid_for_generations(
                diffusion_transformer=diffusion_transformer,
                batch_images=x,  # Using current batch x directly
                epoch=epoch,
                save_img_dir=save_img_dir
            )
            
            # Log FID score to wandb if enabled
            if train_use_wandb and fid_value is not None:
                wandb.log({"fid_score": fid_value, "epoch": epoch})
                
        except Exception as e:
            accelerator.print(f"FID calculation failed at epoch {epoch}: {e}")
    accelerator.end_training()
    accelerator.print("Training complete!")
    if train_use_wandb:
        wandb.finish()



if __name__ == "__main__":
    train_loader = load_dataset("./latents", batch_size=128)
    val_loader = load_dataset("./validation_latents", batch_size=64)

    train(train_loader, val_loader)
