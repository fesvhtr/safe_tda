from utils.dataset_utils import load_embeddings
from utils.model_utils import load_clip
from dataset.tda_patch_dataset import TDAPatchDataset
from tqdm import tqdm
from model.tda_models import *
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import os
import numpy as np
from datetime import datetime

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for vecs, labels in dataloader:
        vecs = vecs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(vecs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu())

    acc, prec, rec, f1 = compute_metrics(labels_all, preds_all)
    return total_loss / len(dataloader), acc, prec, rec, f1


def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for vecs, labels in dataloader:
            vecs = vecs.to(device)
            labels = labels.float().to(device)

            outputs = model(vecs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu())

    acc, prec, rec, f1 = compute_metrics(labels_all, preds_all)
    return total_loss / len(dataloader), acc, prec, rec, f1


def train_loop(train_dataset, val_dataset=None,
               input_dim=400, epochs=10, batch_size=64,
               lr=1e-4, device="cuda", save_path="best_model.pt", modality="text",
               use_wandb=False, wandb_project="safe_tda"):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    model = NSFWPatchMLPClassifierM(input_dim=input_dim).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{time}_{modality}_patch"
    save_path = os.path.join(save_path, f"{run_name}_best.pt")
    if use_wandb:
        wandb.init(project=wandb_project, name=run_name)
        # wandb.watch(model)

    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
              f"Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")

        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/precision": train_prec,
                "train/recall": train_rec,
                "train/f1": train_f1,
                "epoch": epoch
            })

        if val_loader:
            val_loss, val_acc, val_prec, val_rec, val_f1 = eval_model(
                model, val_loader, loss_fn, device)
            print(f"           Val   Loss: {val_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_prec,
                    "val/recall": val_rec,
                    "val/f1": val_f1
                })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                print(f"           ✅ Saved best model to {save_path}")
    return save_path


def evaluate_on_test_set(model, test_loader, device="cuda", use_wandb=False):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import torch.nn.functional as F

    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for vecs, labels in test_loader:
            vecs = vecs.to(device)
            labels = labels.float().to(device)

            outputs = model(vecs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long().cpu()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu())

    acc = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec = recall_score(labels_all, preds_all, zero_division=0)
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    avg_loss = total_loss / len(test_loader)

    print(f"[Test Set] Loss: {avg_loss:.4f} | "
          f"Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    if use_wandb:
        import wandb
        wandb.log({
            "test/loss": avg_loss,
            "test/acc": acc,
            "test/precision": prec,
            "test/recall": rec,
            "test/f1": f1
        })

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


if __name__ == "__main__":
    # --- Keep these initial settings ---
    os.environ["WANDB_API_KEY"] = "da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82" # Temporary environment variable override
    # wandb.login(relogin=True)

    # Set train=True to run the demo training loop
    train = True 
    # Set test=True to run the evaluation on the demo test split
    test = True  
    use_wandb = False
    modality = "text"  # or "image"
    tda_method = ["image"]  # or ["landscape", "image", "betti", "stats"]
    return_mode = "concat"  # or "first"

    model_name = "ViT-L/14"  # or "longclip"
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess, clip_tokenizer = load_clip(model_name, device)
    print("-" * 30)

    # --- Modifications Start Here ---

    # 1. Load ONLY the original 'test' set embeddings
    print("Loading original 'test' set embeddings for demo...")
    safe_embeddings, nsfw_embeddings, _ = load_embeddings(
        clip_model,  clip_preprocess, clip_tokenizer, device, split="test", # Use 'test' split here
        modality=modality
    )
    print("Embeddings loaded.")

    # 2. Create ONE TDAPatchDataset instance using the 'test' data paths
    print("Creating dataset from 'test' data...")
    full_dataset = TDAPatchDataset(
        nsfw_embeddings=nsfw_embeddings,
        # Use the paths corresponding to your original test set
        nsfw_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/test_patch_id_ns75g500.json", 
        safe_embeddings=safe_embeddings,
        safe_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/test_patch_id_ss75g500.json",
        tda_method=tda_method,
        # Use a cache path specific to this demo setup if desired, or keep the test one
        cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_test.pkl",
        plot=False,
        return_mode=return_mode,
    )
    print(f"Full dataset size (from original test set): {len(full_dataset)}")


    # 4. Split the dataset into demo train, validation, and test sets (80/10/10)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size # Ensure all data is used

    print(f"Splitting into: Train={train_size}, Val={val_size}, Test={test_size}")
    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42) 
    demo_train_set, demo_val_set, demo_test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    print("Dataset split complete.")

    # --- Modifications End Here ---

    best_save_path = None # Initialize variable

    if train:
        print("\n--- Starting Demo Training Loop ---")
        # 5. Call train_loop with the demo splits
        best_save_path = train_loop(
            train_dataset=demo_train_set,   # Use demo train set
            val_dataset=demo_val_set,     # Use demo val set
            input_dim=400,                # Keep original parameters or adjust if needed
            epochs=50,                    # Keep original parameters or adjust for demo
            batch_size=128,               # Keep original parameters or adjust for demo
            lr=1e-4,
            device=device,
            use_wandb=use_wandb,
            modality=f"{modality}_demo", # Add demo suffix to modality for wandb/saving
            save_path="/home/muzammal/Projects/safe_proj/safe_tda/data/weights" # Original path, filename includes modality
        )
        print("--- Demo Training Loop Finished ---")


    if test:
        print("\n--- Starting Evaluation on Demo Test Set ---")
        if best_save_path is None or not os.path.exists(best_save_path):
             print(f"Error: Model file not found at {best_save_path}. Cannot run test evaluation.")
             # Optional: Load a default/pre-existing model for testing if training didn't run
             # best_save_path = "path/to/some/existing_demo_model.pt" 
        else:
            # 6. Evaluate on the demo test split
            # Create DataLoader for the demo test set
            demo_test_loader = DataLoader(demo_test_set, batch_size=64) 
            
            # Load the model saved during the demo training
            model = NSFWPatchMLPClassifierM(input_dim=400).to(device)
            print(f"Loading model from: {best_save_path}")
            model.load_state_dict(torch.load(best_save_path, map_location=device)) # Use map_location for flexibility

            evaluate_on_test_set(
                model=model,
                test_loader=demo_test_loader, # Use the demo test loader
                device=device,
                use_wandb=use_wandb,
            )
        print("--- Demo Test Evaluation Finished ---")

    if use_wandb:
        wandb.finish()
    print("Script finished.")
