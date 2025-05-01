from utils.dataset_utils import load_embeddings
from utils.model_utils import load_clip
from dataset.tda_patch_dataset import TDAPatchDataset
from tqdm import tqdm
from model.tda_models import NSFWPatchMLPClassifier
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb


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
               input_dim=300, epochs=10, batch_size=64,
               lr=1e-4, device="cuda", save_path="best_model.pt",
               use_wandb=False, wandb_project="nsfw-patch"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    model = NSFWPatchMLPClassifier(input_dim=input_dim).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_wandb:
        wandb.init(project=wandb_project, name="nsfw-patch-run")
        wandb.watch(model)

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
    model_name = "ViT-L/14"  # or "longclip"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = True
    modality = "text"  # or "image"
    clip_model, clip_preprocess, clip_tokenizer = load_clip(model_name, device)
    print("-" * 30)

    # --- Load the dataset ---
    nsfw_text_embeddings, safe_text_embeddings, _ = load_embeddings(clip_model, clip_tokenizer, device, split="train",
                                                                    modality=modality)

    train_set = TDAPatchDataset(
        nsfw_embeddings=nsfw_text_embeddings,
        nsfw_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\train_patch_id_ns75g10000.json",
        safe_embeddings=safe_text_embeddings,
        safe_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\train_patch_id_ss75g10000.json",
        tda_method="landscape",
        cache_path=r"H:\ProjectsPro\safe_tda\data\cache\text_patch_train.pt",
        plot=False,
        force_recompute=False
    )
    print(f"Total train set size: {len(train_set)}")

    print("Caching train set...")
    for i in tqdm(range(len(train_set)), desc="Train cache"):
        _ = train_set[i]

    # TODO: Add validation set
    nsfw_text_embeddings, safe_text_embeddings, _ = load_embeddings(clip_model, clip_tokenizer, device, split="val",
                                                                    modality=modality)
    val_set = TDAPatchDataset(
        nsfw_embeddings=nsfw_text_embeddings,
        nsfw_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\val_patch_id_ns75g500.json",
        safe_embeddings=safe_text_embeddings,
        safe_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\val_patch_id_ss75g500.json",
        tda_method="landscape",
        cache_path=r"H:\ProjectsPro\safe_tda\data\cache\text_patch_val.pt",
        plot=False,
        force_recompute=False
    )
    print(f"Total val set size: {len(val_set)}")
    print("Caching val set...")
    for i in tqdm(range(len(val_set)), desc="Val cache"):
        _ = val_set[i]

    # 开始训练
    train_loop(
        train_dataset=train_set,
        val_dataset=val_set,
        input_dim=300,
        epochs=100,
        batch_size=64,
        lr=1e-4,
        device="cuda",
        use_wandb=False,
        save_path=r"H:\ProjectsPro\safe_tda\data\weights\nsfw_patch_best.pt"
    )

    test = True
    if test:
        safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_tokenizer, device,
                                                                        split="test", modality=modality)

        test_set = TDAPatchDataset(
            nsfw_embeddings=nsfw_text_embeddings,
            nsfw_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\test_patch_id_ns75g500.json",
            safe_embeddings=safe_text_embeddings,
            safe_group_indices_path=r"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\test_patch_id_ss75g500.json",
            tda_method="landscape",
            cache_path=r"H:\ProjectsPro\safe_tda\data\cache\text_patch_test.pt",
            plot=False,
            force_recompute=False
        )
        print(f"Total dataset size: {len(test_set)}")
        for _ in tqdm(range(len(test_set)), desc="Test cache"):
            _ = test_set[_]
        test_loader = DataLoader(test_set, batch_size=64)
        model = NSFWPatchMLPClassifier(input_dim=300).to(device)
        model.load_state_dict(torch.load(r"H:\ProjectsPro\safe_tda\data\weights\nsfw_patch_best.pt"))
        evaluate_on_test_set(
            model=model,
            test_loader=test_loader,
            device=device,
            use_wandb=False
        )
