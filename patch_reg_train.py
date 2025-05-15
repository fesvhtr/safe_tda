from utils.dataset_utils import load_embeddings
from utils.model_utils import load_clip
from dataset.tda_patch_dataset import TDAPatchRegDataset
from tqdm import tqdm
from model.tda_models import *
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb
import os
import numpy as np
from datetime import datetime

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    preds_all, labels_all = [], []

    for vecs, labels in dataloader:
        vecs = vecs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(vecs).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend(outputs.detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    mse, mae, rmse, r2 = compute_metrics(labels_all, preds_all)
    return total_loss / len(dataloader), mse, mae, rmse, r2


def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for vecs, labels in dataloader:
            vecs = vecs.to(device)
            labels = labels.float().to(device)

            outputs = model(vecs).squeeze()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds_all.extend(outputs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    mse, mae, rmse, r2 = compute_metrics(labels_all, preds_all)
    return total_loss / len(dataloader), mse, mae, rmse, r2


def train_loop(train_dataset, val_dataset=None,
               input_dim=850, epochs=10, batch_size=64,
               lr=1e-4, device="cuda:3", save_path="best_model.pt", modality="text",
               use_wandb=False, wandb_project="safe_tda_re"):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    print("loading data done")
    model = NSFWPatchMLPClassifierL(input_dim=input_dim).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{time}_{modality}_reg_patch"
    save_path = os.path.join(save_path, f"{run_name}_reg_best.pt")
    if use_wandb:
        wandb.init(project=wandb_project, name=run_name)

    best_val_r2 = -float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_mse, train_mae, train_rmse, train_r2 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
              f"MSE: {train_mse:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | R2: {train_r2:.4f}")

        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/mse": train_mse,
                "train/mae": train_mae,
                "train/rmse": train_rmse,
                "train/r2": train_r2,
                "epoch": epoch
            })

        if val_loader:
            val_loss, val_mse, val_mae, val_rmse, val_r2 = eval_model(
                model, val_loader, loss_fn, device)
            print(f"           Val   Loss: {val_loss:.4f} | "
                  f"MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/mse": val_mse,
                    "val/mae": val_mae,
                    "val/rmse": val_rmse,
                    "val/r2": val_r2
                })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), save_path)
                print(f"           ✅ Saved best model to {save_path}")
    return save_path


def evaluate_on_test_set(model, test_loader, device="cuda:3", use_wandb=False):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0.0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for vecs, labels in test_loader:
            vecs = vecs.to(device)
            labels = labels.float().to(device)

            outputs = model(vecs).squeeze()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds_all.extend(outputs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    mse = mean_squared_error(labels_all, preds_all)
    mae = mean_absolute_error(labels_all, preds_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels_all, preds_all)
    avg_loss = total_loss / len(test_loader)

    print(f"[Test Set] Loss: {avg_loss:.4f} | "
          f"MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    if use_wandb:
        wandb.log({
            "test/loss": avg_loss,
            "test/mse": mse,
            "test/mae": mae,
            "test/rmse": rmse,
            "test/r2": r2
        })

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }




if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82"

    test = True
    train = True
    use_wandb = True
    hybrid_train = True
    hybrid_test = False
    modality = "text"
    tda_method = ["landscape", "image", "betti"] # ["landscape", "image", "betti", "stats"]
    return_mode = "concat"
    id_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/"
    model_name = "ViT-L/14"
    device = "cuda:2" if torch.cuda.is_available() else "cpu"


    # print(f"Using device: {device}")
    # clip_model, clip_preprocess, clip_tokenizer = load_clip(model_name, device)
    # print("-" * 30)

    if train:
        # --- Load the dataset ---
        # safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer,  device, split="train",
        #                                                                 modality=modality)
        safe_text_embeddings, nsfw_text_embeddings = None, None
        if hybrid_train:
            # train set Hybrid
            train_set = TDAPatchRegDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                safe_embeddings=safe_text_embeddings,
                mix_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/train_patch_id_mix50-100g10000.json",
                tda_method=tda_method,
                return_mode=return_mode,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_reg_patch_train_hy.pkl",
                plot=False,
                force_recompute=False
            )
        
        else:
            # train set N=75
            train_set = TDAPatchRegDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                safe_embeddings=safe_text_embeddings,
                mix_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/train_patch_id_mix75g10000.json",
                tda_method=tda_method,
                return_mode=return_mode,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_reg_patch_train.pkl",
                plot=False,
                force_recompute=False
            )
        
        print(f"Total train set size: {len(train_set)}")

        # safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model,clip_preprocess, clip_tokenizer, device, split="val",
        #                                                                 modality=modality)
        safe_text_embeddings, nsfw_text_embeddings = None, None
        if hybrid_test:
            pass
        else:
            val_set = TDAPatchRegDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                safe_embeddings=safe_text_embeddings,
                mix_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/val_patch_id_mix75g1000.json",
                tda_method=tda_method,
                return_mode=return_mode,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_reg_patch_val.pkl",
                plot=False,
                force_recompute=False
            )
        print(f"Total val set size: {len(val_set)}")


        best_save_path = train_loop(
            train_dataset=train_set,
            val_dataset=val_set,
            input_dim=850,
            epochs=200,
            batch_size=128,
            lr=1e-5,
            device="cuda",
            use_wandb=use_wandb,
            modality=modality,
            save_path="/home/muzammal/Projects/safe_proj/safe_tda/data/weights"
        )

    if test:
        # safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer, device,
        #                                                                 split="test", modality=modality)
        safe_text_embeddings, nsfw_text_embeddings = None, None
        if hybrid_test:
            pass
        else:
            test_set = TDAPatchRegDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                safe_embeddings=safe_text_embeddings,
                mix_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/test_patch_id_mix75g1000.json",
                tda_method=tda_method,
                return_mode=return_mode,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_reg_patch_test.pkl",
                plot=False,
                force_recompute=False
            )
        test_loader = DataLoader(test_set, batch_size=64)
        model = NSFWPatchMLPClassifierL(input_dim=850).to(device)
        model.load_state_dict(torch.load(best_save_path))
        evaluate_on_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            use_wandb=use_wandb,
        )
    wandb.finish()
