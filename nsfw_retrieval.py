import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import clip
from torch.utils.data import Dataset, DataLoader
from model import longclip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(json_file, safe_img_dir, nsfw_img_dir):
    """
    Load the dataset from the JSON file and prepare data for retrieval.
    
    Args:
        json_file: Path to the ViSU-Text JSON file
        safe_img_dir: Directory containing COCO images
        
    Returns:
        safe_texts: List of safe text descriptions
        safe_image_paths: List of paths to corresponding safe images
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    safe_texts = []
    safe_image_paths = []
    nsfw_texts = []
    nsfw_image_paths = []
    
    for item in data:
        safe_text = item["safe"]
        nsfw_text = item["nsfw"]
        coco_id = item["coco_id"]
        nsfw_id = "flux_" + str(item["incremental_id"])
        safe_img_path = os.path.join(safe_img_dir, f"{coco_id}.jpg")
        nsfw_img_path = os.path.join(nsfw_img_dir, f"{nsfw_id}.png")
        if os.path.exists(safe_img_path):
            safe_texts.append(safe_text)
            safe_image_paths.append(safe_img_path)
        if os.path.exists(nsfw_img_path):
            nsfw_texts.append(nsfw_text)
            nsfw_image_paths.append(nsfw_img_path)
    print(f"Loaded {len(safe_texts)} safe texts and {len(nsfw_texts)} NSFW texts")
    print(f"Loaded {len(safe_image_paths)} safe images and {len(nsfw_image_paths)} NSFW images")
    
    return safe_texts, safe_image_paths, nsfw_texts, nsfw_image_paths

def load_dataset_with_pairs(json_file, safe_img_dir, nsfw_img_dir):
    """
    Load the dataset from the JSON file with paired safe/NSFW content.
    
    Args:
        json_file: Path to the ViSU-Text JSON file
        safe_img_dir: Directory containing safe images
        nsfw_img_dir: Directory containing NSFW images
        
    Returns:
        pairs: List of dictionaries containing paired safe/NSFW content
        all_texts: List of all texts (safe + NSFW)
        all_image_paths: List of all image paths (safe + NSFW)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pairs = []
    all_texts = []
    all_image_paths = []
    
    for item in data:
        safe_text = item["safe"]
        nsfw_text = item["nsfw"]
        coco_id = item["coco_id"]
        nsfw_id = "flux_" + str(item["incremental_id"])
        
        safe_img_path = os.path.join(safe_img_dir, f"{coco_id}.jpg")
        nsfw_img_path = os.path.join(nsfw_img_dir, f"{nsfw_id}.png")
        
        # Only include if both images exist
        if os.path.exists(safe_img_path) and os.path.exists(nsfw_img_path):
            pairs.append({
                "safe_text": safe_text,
                "nsfw_text": nsfw_text,
                "safe_img_path": safe_img_path,
                "nsfw_img_path": nsfw_img_path
            })
            all_texts.extend([safe_text, nsfw_text])
            all_image_paths.extend([safe_img_path, nsfw_img_path])
    
    print(f"Loaded {len(pairs)} paired safe/NSFW items")
    print(f"Total texts: {len(all_texts)}, Total images: {len(all_image_paths)}")
    
    return pairs, all_texts, all_image_paths

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.tokenizer([self.texts[idx]])

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self.preprocess(image)

def text_to_image_retrieval(model, preprocess, tokenizer, texts, images, k_values=[1, 5, 20], batch_size=32):
    """
    Perform text-to-image retrieval and calculate R@K metrics with batch processing.
    """
    # Extract text features in batches
    text_dataset = TextDataset(texts, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
    
    text_features = []
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc="Extracting text features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_text(batch)
            text_features.append(batch_features)
     
    text_features = torch.cat(text_features).to(device)
    text_features = F.normalize(text_features, dim=1)
    
    # Extract image features in batches
    image_dataset = ImageDataset(images, preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
    
    image_features = []
    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Extracting image features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_image(batch)
            image_features.append(batch_features)
            
    image_features = torch.cat(image_features).to(device)
    image_features = F.normalize(image_features, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.matmul(text_features, image_features.t()).cpu().numpy()
    
    # Calculate R@K metrics
    metrics = {}
    for k in k_values:
        recall = 0
        for i in range(len(texts)):
            # For each text, find the top k most similar images
            top_indices = np.argsort(-similarity[i])[:k]
            # Check if the correct image (with the same index) is in top k
            if i in top_indices:
                recall += 1
        
        metrics[f"R@{k}"] = recall / len(texts) * 100.0
    
    return metrics

def image_to_text_retrieval(model, preprocess, tokenizer, images, texts, k_values=[1, 5, 20], batch_size=32):
    """
    Perform image-to-text retrieval and calculate R@K metrics with batch processing.
    """
    # Extract image features in batches
    image_dataset = ImageDataset(images, preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
    
    image_features = []
    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Extracting image features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_image(batch)
            image_features.append(batch_features)
            
    image_features = torch.cat(image_features).to(device)
    image_features = F.normalize(image_features, dim=1)
    
    # Extract text features in batches
    text_dataset = TextDataset(texts, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
    
    text_features = []
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc="Extracting text features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_text(batch)
            text_features.append(batch_features)
    
    text_features = torch.cat(text_features).to(device)
    text_features = F.normalize(text_features, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.matmul(image_features, text_features.t()).cpu().numpy()
    
    # Calculate R@K metrics
    metrics = {}
    for k in k_values:
        recall = 0
        for i in range(len(images)):
            # For each image, find the top k most similar texts
            top_indices = np.argsort(-similarity[i])[:k]
            # Check if the correct text (with the same index) is in top k
            if i in top_indices:
                recall += 1
        
        metrics[f"R@{k}"] = recall / len(images) * 100.0
    
    return metrics

def nsfw_text_to_image_retrieval(model, preprocess, tokenizer, pairs, batch_size=32, k_values=[1, 5, 20]):
    """
    Perform NSFW text-to-image retrieval where NSFW texts are used to retrieve safe images.
    
    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        pairs: List of dictionaries containing paired safe/NSFW content
        batch_size: Batch size for processing
        k_values: List of K values for R@K calculation
    
    Returns:
        Dictionary of R@K metrics
    """
    # Extract query texts (NSFW texts)
    nsfw_texts = [pair["nsfw_text"] for pair in pairs]
    
    # Create the database of all images (both safe and NSFW)
    all_image_paths = []
    for pair in pairs:
        all_image_paths.extend([pair["safe_img_path"], pair["nsfw_img_path"]])
    
    # Extract text features in batches
    text_dataset = TextDataset(nsfw_texts, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
    
    text_features = []
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc="Extracting NSFW text features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_text(batch)
            text_features.append(batch_features)
    
    text_features = torch.cat(text_features).to(device)
    text_features = F.normalize(text_features, dim=1)
    
    # Extract image features in batches
    image_dataset = ImageDataset(all_image_paths, preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
    
    image_features = []
    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Extracting all image features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_image(batch)
            image_features.append(batch_features)
            
    image_features = torch.cat(image_features).to(device)
    image_features = F.normalize(image_features, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.matmul(text_features, image_features.t()).cpu().numpy()
    
    # Calculate R@K metrics (NSFW text should retrieve the corresponding safe image)
    metrics = {}
    for k in k_values:
        recall = 0
        for i, pair in enumerate(pairs):
            # For each NSFW text, find the index of its corresponding safe image in the all_image_paths list
            safe_img_index = all_image_paths.index(pair["safe_img_path"])
            
            # Find the top k most similar images
            top_indices = np.argsort(-similarity[i])[:k]
            
            # Check if the correct safe image is in top k
            if safe_img_index in top_indices:
                recall += 1
        
        metrics[f"R@{k}"] = recall / len(pairs) * 100.0
    
    return metrics

def nsfw_image_to_text_retrieval(model, preprocess, tokenizer, pairs, batch_size=32, k_values=[1, 5, 20]):
    """
    Perform NSFW image-to-text retrieval where NSFW images are used to retrieve safe texts.
    
    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        tokenizer: Tokenizer for processing texts
        pairs: List of dictionaries containing paired safe/NSFW content
        batch_size: Batch size for processing
        k_values: List of K values for R@K calculation
    
    Returns:
        Dictionary of R@K metrics
    """
    # Extract query images (NSFW images)
    nsfw_image_paths = [pair["nsfw_img_path"] for pair in pairs]
    
    # Create the database of all texts (both safe and NSFW)
    all_texts = []
    for pair in pairs:
        all_texts.extend([pair["safe_text"], pair["nsfw_text"]])
    
    # Extract image features in batches
    image_dataset = ImageDataset(nsfw_image_paths, preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
    
    image_features = []
    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Extracting NSFW image features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_image(batch)
            image_features.append(batch_features)
            
    image_features = torch.cat(image_features).to(device)
    image_features = F.normalize(image_features, dim=1)
    
    # Extract text features in batches
    text_dataset = TextDataset(all_texts, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
    
    text_features = []
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc="Extracting all text features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = model.encode_text(batch)
            text_features.append(batch_features)
    
    text_features = torch.cat(text_features).to(device)
    text_features = F.normalize(text_features, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.matmul(image_features, text_features.t()).cpu().numpy()
    
    # Calculate R@K metrics (NSFW image should retrieve the corresponding safe text)
    metrics = {}
    for k in k_values:
        recall = 0
        for i, pair in enumerate(pairs):
            # For each NSFW image, find the index of its corresponding safe text in the all_texts list
            safe_text_index = all_texts.index(pair["safe_text"])
            
            # Find the top k most similar texts
            top_indices = np.argsort(-similarity[i])[:k]
            
            # Check if the correct safe text is in top k
            if safe_text_index in top_indices:
                recall += 1
        
        metrics[f"R@{k}"] = recall / len(pairs) * 100.0
    
    return metrics

def safe_retrieval(model, preprocess, tokenizer, safe_texts, safe_image_paths, batch_size=32):
    # Text-to-image retrieval
    print("\nPerforming text-to-image retrieval evaluation...")
    t2i_metrics = text_to_image_retrieval(model, preprocess, tokenizer, safe_texts, safe_image_paths, batch_size=batch_size)

    print("\nText-to-Image Retrieval Results:")
    for k, v in t2i_metrics.items():
        print(f"{k}: {v:.2f}%")
    
    # Image-to-text retrieval
    print("\nPerforming image-to-text retrieval evaluation...")
    i2t_metrics = image_to_text_retrieval(model, preprocess, tokenizer, safe_image_paths, safe_texts, batch_size=batch_size)
    
    print("\nImage-to-Text Retrieval Results:")
    for k, v in i2t_metrics.items():
        print(f"{k}: {v:.2f}%")
    return t2i_metrics, i2t_metrics

def nsfw_retrieval(model, preprocess, tokenizer, pairs, batch_size=32):
    # NSFW text-to-image retrieval (NSFW texts retrieving safe images)
    print("\nPerforming NSFW text-to-safe image retrieval evaluation...")
    nsfw_t2i_metrics = nsfw_text_to_image_retrieval(model, preprocess, tokenizer, pairs, batch_size=batch_size)
    print("\nNSFW Text-to-Safe Image Retrieval Results:")
    for k, v in nsfw_t2i_metrics.items():
        print(f"{k}: {v:.2f}%")
    
    # NSFW image-to-text retrieval (NSFW images retrieving safe texts)
    print("\nPerforming NSFW image-to-safe text retrieval evaluation...")
    nsfw_i2t_metrics = nsfw_image_to_text_retrieval(model, preprocess, tokenizer, pairs, batch_size=batch_size)
    print("\nNSFW Image-to-Safe Text Retrieval Results:")
    for k, v in nsfw_i2t_metrics.items():
        print(f"{k}: {v:.2f}%")

def main():
    """Main function to run the evaluation."""
    # Set paths
    json_file = "/home/muzammal/Projects/safe_proj/long_safe_clip/dataset/visu/ViSU-Text_test.json"
    safe_img_dir = "/home/muzammal/Projects/safe_proj/long_safe_clip/dataset/visu/image/test_coco"  # Update with actual path
    unsafe_image_dir = "/home/muzammal/Projects/safe_proj/long_safe_clip/dataset/visu/image/test_nsfw"  # Update with actual path
    model_name = "longclip"  # or "longclip"
    task = ["nsfw_retrieval"]  # Specify the tasks to run

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load data
    print("Loading dataset...")
    safe_texts, safe_image_paths, nsfw_texts, unsafe_image_paths = load_dataset(json_file, safe_img_dir, unsafe_image_dir)
    # Load paired data for NSFW evaluations
    print("Loading paired dataset...")
    pairs, all_texts, all_image_paths = load_dataset_with_pairs(json_file, safe_img_dir, unsafe_image_dir)
    
    # Set batch size based on GPU memory
    batch_size = 64  # Adjust based on your GPU memory
    
    # Load the model
    if model_name == "longclip":
        print("Loading Long-CLIP model...")
        model, preprocess = longclip.load("/home/muzammal/Projects/safe_proj/long_safe_clip/weights/longclip-L.pt", device=device)
        tokenizer = longclip.tokenize
    elif model_name == "ViT-L/14":
        print("Loading CLIP model...")
        model, preprocess = clip.load(model_name, device=device)
        tokenizer = clip.tokenize

    model.eval()
    
    if 'safe_retrieval' in task:
        safe_retrieval(model, preprocess, tokenizer, safe_texts, safe_image_paths, batch_size=batch_size)
    if 'nsfw_retrieval' in task:
        nsfw_retrieval(model, preprocess, tokenizer, pairs, batch_size=batch_size)
    


if __name__ == "__main__":
    main()
