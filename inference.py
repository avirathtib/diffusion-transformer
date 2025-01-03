import torch
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from diffusers import AutoencoderKL
import numpy as np
from tqdm import tqdm
import os
import torchvision.utils as vutils
from models.denoiser import Denoiser
import matplotlib.pyplot as plt

to_pil = transforms.ToPILImage()

class DiffusionReverseProcess:
    def __init__(self, model, vae, device, model_dtype):
        self.model = model
        self.vae = vae
        self.device = device
        self.model_dtype = model_dtype

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                num_imgs, 4, img_size, img_size,
                dtype=self.model_dtype, device=self.device,
                generator=generator
            )
        else:
            return seeds.to(self.device, self.model_dtype)

    def pred_image(self, noisy_image, labels, noise_level, class_guidance):
        num_imgs = noisy_image.size(0)
        noises = torch.full((2 * num_imgs, 1), noise_level)
        x0_pred = self.model(
            torch.cat([noisy_image, noisy_image]),
            noises.to(self.device, self.model_dtype),
            labels.to(self.device, self.model_dtype),
        )
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

    @torch.no_grad()
    def generate(self, labels, n_iter, class_guidance = 3, image_size = 32, num_imgs = 16, sharp_f = 0.1, bright_f = 0.1, seed = 10, exponent = 1, seeds = None, noise_levels = None, use_ddpm_plus = False):
        if noise_levels is None:
                noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99
        x_t = self.initialize_image(seeds, num_imgs, image_size, seed)
        labels = labels.unsqueeze(0)
        labels = torch.cat([labels, torch.zeros_like(labels)], dim = 0)
        print(labels.shape, "lol")
        self.model.eval()
        x0_pred_prev = None
        for i in tqdm(range(len(noise_levels) - 1)):
                curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

                x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance)

                if x0_pred_prev is None:
                    x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
                else:
                    if use_ddpm_plus:
                        # x0_pred is a combination of the two previous x0_pred:
                        D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                    else:
                        # ddim:
                        D = x0_pred

                    x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

                x0_pred_prev = x0_pred
                if i % 10 == 0:
                    plt.imshow(to_pil((x_t[0].cpu() + 1) / 2))  # Normalize to [0, 1] for display
                    plt.axis('off')
                    plt.show()

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance)
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred / 0.18215).to(self.model_dtype))[0].cpu()
        return x0_pred_img, x0_pred


def load_clip_text_encoder(model_name="sentence-transformers/sentence-t5-xxl"):
    return SentenceTransformer(model_name)

def encode_text(clip_model, text):
    if isinstance(text, str):
        text = [text]
    return clip_model.encode(text, convert_to_tensor=True, show_progress_bar=False)[0]


class DiffusionTransformer:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.load_denoiser_model()
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)
        self.clip_model = load_clip_text_encoder("sentence-transformers/sentence-t5-xxl").to(device)
        self.diffuser = DiffusionReverseProcess(self.model, self.vae, device, torch.float32)

    def load_denoiser_model(self):
        checkpoint_dir = "checkpoints"
        checkpoint_files = sorted(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
            key=os.path.getmtime,
        )

        model = Denoiser(
            image_size=32,
            noise_embed_dims=512,
            patch_size=2,
            embed_dim=640,
            dropout=0.2,
            n_layers=10
        ).to(self.device)

        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"Loading model from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint.get("model_ema", checkpoint))
        else:
            print("No checkpoint found. Initializing model with random weights.")

        model.eval()
        return model


    def generate_image_from_text(self, prompt, class_guidance=11, seed=11, num_imgs=1, img_size=32, n_iter=150):
        nrow = int(np.sqrt(num_imgs))
        labels = encode_text(self.clip_model, [prompt] * num_imgs)
        out_img, out_latent = self.diffuser.generate(labels, n_iter, class_guidance, img_size, num_imgs, seed=seed)
        out = to_pil((vutils.make_grid((out_img + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
        return out
