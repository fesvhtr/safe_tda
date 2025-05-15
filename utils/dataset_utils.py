import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer([self.texts[idx]], truncate=True)


class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self.preprocess(image)
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
    print("Loading dataset...")
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
        nsfw_id = "nsfw_" + str(item["incremental_id"])
        safe_img_path = os.path.join(safe_img_dir, f"{coco_id}.jpg")
        nsfw_img_path = os.path.join(nsfw_img_dir, f"{nsfw_id}.png")

        safe_texts.append(safe_text)
        if os.path.exists(safe_img_path):
            safe_image_paths.append(safe_img_path)
        else:
            # TODO: add warning
            pass

        nsfw_texts.append(nsfw_text)
        if os.path.exists(nsfw_img_path):
            nsfw_image_paths.append(nsfw_img_path)
    if len(safe_texts) != len(safe_image_paths):
        print(f"Warning: safe_text has no corresponding image.")
    if len(nsfw_texts) != len(nsfw_image_paths):
        print(f"Warning: nsfw_text has no corresponding image.")
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
    print("Loading paired dataset...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    pairs = []
    all_texts = []
    all_image_paths = []

    for item in data:
        safe_text = item["safe"]
        nsfw_text = item["nsfw"]
        coco_id = item["coco_id"]
        nsfw_id = "nsfw_" + str(item["incremental_id"])

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


def extract_text_embeddings(texts, clip_model, tokenizer, device="cuda", batch_size=32):
    print("Extracting TEXT embeddings...")
    text_dataset = TextDataset(texts, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
    text_features = []
    with torch.no_grad():
        for batch in tqdm(text_dataloader, desc="Extracting text features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = clip_model.encode_text(batch)
            text_features.append(batch_features)

    text_features = torch.cat(text_features).to(device)
    text_features = F.normalize(text_features, dim=1)
    print(f"text_features shape: {text_features.shape}")
    return text_features.cpu().numpy()


def extract_image_embeddings(image_paths, clip_model, preprocess, device="cuda", batch_size=16):
    image_dataset = ImageDataset(image_paths, preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
    image_features = []
    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Extracting image features"):
            batch = batch.to(device)
            batch = batch.squeeze(1)
            batch_features = clip_model.encode_image(batch)
            image_features.append(batch_features)

    image_features = torch.cat(image_features).to(device)
    image_features = F.normalize(image_features, dim=1)
    print(f"image_features shape: {image_features.shape}")
    return image_features.cpu().numpy()


def load_embeddings(clip_model,  clip_preprocess, clip_tokenizer, device, split="test", modality="text"):
    # load the raw dataset
    if split == "train5k":
        json_file = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_train_5k.json"
        safe_img_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_coco_5k"
        nsfw_image_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored_5k"
    elif split == "traink":
        json_file = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_train_5k.json"
        safe_img_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_coco_5k"
        nsfw_image_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored_5k"
    elif split == "train":
        json_file = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_train.json"
        safe_img_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_coco"
        nsfw_image_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored"
    elif split == "test":
        json_file = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_test.json"
        safe_img_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/test_coco"
        nsfw_image_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/test_FLUX_Unsensored"
    elif split == "val":
        json_file = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_validation.json"
        safe_img_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/val_coco"
        nsfw_image_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/val_FLUX_Unsensored"

    safe_texts, safe_image_paths, nsfw_texts, nsfw_image_paths = load_dataset(json_file, safe_img_dir,
                                                                                  nsfw_image_dir)
    if modality == "text":
        safe_text_embeddings = extract_text_embeddings(safe_texts, clip_model, clip_tokenizer, device, batch_size=128)
        nsfw_text_embeddings = extract_text_embeddings(nsfw_texts, clip_model, clip_tokenizer, device, batch_size=128)
        analysis_label = "Text"
        return safe_text_embeddings, nsfw_text_embeddings, analysis_label
    elif modality == "image":
        safe_image_embeddings = extract_image_embeddings(safe_image_paths, clip_model, clip_preprocess, device,
                                                          batch_size=32)
        nsfw_image_embeddings = extract_image_embeddings(nsfw_image_paths, clip_model, clip_preprocess, device,
                                                         batch_size=32)
        analysis_label = "Image"
        return safe_image_embeddings, nsfw_image_embeddings, analysis_label