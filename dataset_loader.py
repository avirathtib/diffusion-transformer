import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def load_dataset(latent_folder, batch_size=64):
    latent_data = []
    label_embeddings = []
    file_count = 0  # Counter to track number of files loaded

    for latent_file in os.listdir(latent_folder):
        file_path = os.path.join(latent_folder, latent_file)
        if file_path.endswith(".pt"):
            try:
                data = torch.load(file_path)
                latent = data["latent"]
                text_embedding = data["text_embedding"]

                latent_data.append(latent)
                label_embeddings.append(text_embedding)
                
                file_count += 1  # Increment counter
                print(f"Loaded: {latent_file} ({file_count} files)")
                
            except Exception as e:
                print(f"Error loading {latent_file}: {e}")
    
    print(f"Total files loaded: {file_count}")
    
    latent_data = torch.stack(latent_data)
    label_embeddings = torch.stack(label_embeddings)

    dataset = TensorDataset(latent_data, label_embeddings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
