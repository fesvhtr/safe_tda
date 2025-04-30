from clip import clip
from model import longclip


def load_clip(model_name="ViT-L/14", device="cuda"):
    if model_name == "longclip":
        print("Loading Long-CLIP model...")
        model, preprocess = longclip.load("/home/muzammal/Projects/safe_proj/long_safe_clip/weights/longclip-L.pt", device=device)
        tokenizer = longclip.tokenize
    elif model_name == "ViT-L/14":
        print("Loading CLIP model...")
        model, preprocess = clip.load(model_name, device=device)
        tokenizer = clip.tokenize
    return model, preprocess, tokenizer
