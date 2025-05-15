from utils.dataset_utils import load_embeddings
from utils.model_utils import load_clip
from dataset.tda_patch_dataset import TDAPatchClsDataset
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
               input_dim=700, epochs=10, batch_size=64,
               lr=1e-4, device="cuda:3", save_path="best_model.pt", modality="text",
               use_wandb=False, wandb_project="safe_tda"):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    print("loading data done")
    model = NSFWPatchMLPClassifierL(input_dim=input_dim).to(device)
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


def evaluate_on_test_set(model, test_loader, device="cuda:3", use_wandb=False):
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
    os.environ["WANDB_API_KEY"] = "da3ef2608ceaa362d6e40d1d92b4e4e6ebbe9f82"     # 临时覆盖环境变量
    # wandb.login(relogin=True)


    test = True
    train = True
    use_wandb = True
    hybrid_train = True
    hybrid_test = False
    modality = "text"  # or "image"
    tda_method = ["landscape","image"]
    return_mode = "concat"
    id_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/"

    model_name = "ViT-L/14"  # or "longclip"
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    clip_model, clip_preprocess, clip_tokenizer = load_clip(model_name, device)
    print("-" * 30)

    if train:
        # --- Load the dataset ---
        safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer,  device, split="train",
                                                                        modality=modality)
        if hybrid_train:
            # train set Hybrid
            train_set = TDAPatchClsDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                nsfw_group_indices_path= os.path.join(id_dir, "train_patch_id_ns50-100g5000.json"),
                safe_embeddings=safe_text_embeddings,
                safe_group_indices_path= os.path.join(id_dir, "train_patch_id_ss50-100g5000.json"),
                tda_method=tda_method,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_train_hy.pkl",
                plot=False,
                return_mode=return_mode,
            )
        else:
            # train set N=75
            train_set = TDAPatchClsDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                nsfw_group_indices_path= os.path.join(id_dir, "train_patch_id_ns75g5000.json"),
                safe_embeddings=safe_text_embeddings,
                safe_group_indices_path= os.path.join(id_dir, "train_patch_id_ss75g5000.json"),
                tda_method=tda_method,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_train.pkl",
                plot=False,
                return_mode=return_mode,
            )
        
        print(f"Total train set size: {len(train_set)}")

        safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model,clip_preprocess, clip_tokenizer, device, split="val",
                                                                        modality=modality)
        if hybrid_test:
            pass
        else:
            val_set = TDAPatchClsDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                nsfw_group_indices_path=os.path.join(id_dir, "val_patch_id_ns75g500.json"),
                safe_embeddings=safe_text_embeddings,
                safe_group_indices_path=os.path.join(id_dir, "val_patch_id_ss75g500.json"),
                tda_method=tda_method,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_val.pkl",
                plot=False,
                return_mode=return_mode,
            )
        print(f"Total val set size: {len(val_set)}")


        best_save_path = train_loop(
            train_dataset=train_set,
            val_dataset=val_set,
            input_dim=700,
            epochs=100,
            batch_size=128,
            lr=1e-5,
            device="cuda",
            use_wandb=use_wandb,
            modality=modality,
            save_path="/home/muzammal/Projects/safe_proj/safe_tda/data/weights"
        )

    if test:
        safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer, device,
                                                                        split="test", modality=modality)
        if hybrid_test:
            pass
        else:
            test_set = TDAPatchClsDataset(
                nsfw_embeddings=nsfw_text_embeddings,
                nsfw_group_indices_path=os.path.join(id_dir, "test_patch_id_ns75g500.json"),
                safe_embeddings=safe_text_embeddings,
                safe_group_indices_path=os.path.join(id_dir, "test_patch_id_ss75g500.json"),
                tda_method=tda_method,
                cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_test.pkl",
                plot=False,
                return_mode=return_mode,
            )
        test_loader = DataLoader(test_set, batch_size=64)
        model = NSFWPatchMLPClassifierL(input_dim=700).to(device)
        model.load_state_dict(torch.load(best_save_path))
        evaluate_on_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            use_wandb=use_wandb,
        )
    wandb.finish()
