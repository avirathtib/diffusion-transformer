# Model Hyperparameters
denoiser_image_size = 32                # Size of the input images
denoiser_noise_embed_dims = 512         # Dimensionality of noise embeddings
denoiser_patch_size = 2                 # Patch size for the PatchEmbedding module
denoiser_embed_dim = 640                # Embedding dimensionality in the model
denoiser_dropout = 0.2                  # Dropout probability
denoiser_n_layers = 10                   # Number of layers in the Denoiser model
denoiser_text_emb_size = 768            # Size of the text embeddings (e.g., CLIP embeddings)
denoiser_n_channels = 4                 # Number of input/output channels
denoiser_mlp_multiplier = 6  

# Training Parameters
train_batch_size = 96                  # Batch size for training
train_lr = 5e-5                         # Learning rate
train_n_epoch = 100                     # Number of epochs
train_alpha = 0.999                     # EMA smoothing factor
train_beta_a = 1                        # Alpha parameter for Beta distribution for noise levels
train_beta_b = 2.5                      # Beta parameter for Beta distribution for noise levels
train_save_and_eval_every_iters = 1000  # Save and evaluate checkpoints every X iterations
train_compile = False                   # Whether to use PyTorch 2.0 compile for optimization
train_use_wandb = True                  # Whether to use WandB for logging and tracking
train_from_scratch = True               # Whether to train from scratch or resume training

# # WandB Settings
# wandb_project_name = "Denoiser-Training"  # Name of your WandB project
# wandb_entity = ""                       # Set this to your WandB username/team name if needed
# train_run_id = ""       # Placeholder for resuming specific runs (if needed)
# train_model_name = ""        # Default checkpoint filename

# # Dataset Parameters
# latent_folder = ""              # Path to the folder containing latent dataset files
