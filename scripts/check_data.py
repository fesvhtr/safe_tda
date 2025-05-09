from utils.dataset_utils import load_embeddings
from utils.model_utils import load_clip
from dataset.tda_patch_dataset import TDAPatchDataset
from tqdm import tqdm
import torch.nn.functional as F
from model.tda_models import *
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import os
import numpy as np
from datetime import datetime
import random
modality = "text"  # or "image"
tda_method = ["landscape"]

model_name = "ViT-L/14"  # or "longclip"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess, clip_tokenizer = load_clip(model_name, device)
print("-" * 30)
safe_text_embeddings, nsfw_text_embeddings, _ = load_embeddings(clip_model, clip_tokenizer, device, split="val",
                                                                modality=modality)
val_set = TDAPatchDataset(
    nsfw_embeddings=nsfw_text_embeddings,
    nsfw_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/val_patch_id_ns75g500.json",
    safe_embeddings=safe_text_embeddings,
    safe_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/val_patch_id_ss75g500.json",
    tda_method=tda_method,
    cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{modality}_patch_val.pkl",
    plot=False,
    force_recompute=False
)
dataset_size = len(val_set)
print(f"Total dataset size: {dataset_size}")

# --- 随机抽取并存储向量和标签 ---
fetched_vectors = []
fetched_indices = []
fetched_labels = [] # <-- 存储标签以供后续分析
num_samples_to_check = 20  # <-- 可以增加样本数量以便更好地分析

if dataset_size > 0:
    print(f"\nFetching {num_samples_to_check} random samples to check vector differences:")
    for _ in tqdm(range(num_samples_to_check), desc="Fetching random samples"):
        random_index = random.randint(0, dataset_size - 1)
        try:
            vec, label = val_set[random_index]
            print(f"  Index: {random_index}, Label: {label}")
            print(f"  Vector Sum: {torch.sum(vec).item()}") # 检查和是否为0
            # 确保向量是浮点类型以便计算
            fetched_vectors.append(vec.float())
            fetched_indices.append(random_index)
            fetched_labels.append(label) # <-- 存储标签
        except Exception as e:
            print(f"Error fetching sample at random index {random_index}: {e}")
            continue

    # --- 使用距离/相似度指标检查差异 ---
    if len(fetched_vectors) > 1:
        print(f"\n--- Checking differences between the {len(fetched_vectors)} fetched vectors ---")

        distances_l2 = []
        cosine_similarities = []

        # 将第一个向量作为比较基准
        first_vector = fetched_vectors[0]
        first_index = fetched_indices[0]
        first_label = fetched_labels[0]
        print(f"Comparing against vector from index: {first_index} (Label: {first_label})")

        # 计算与其他向量的距离和相似度
        for i in range(1, len(fetched_vectors)):
            current_vector = fetched_vectors[i]
            current_index = fetched_indices[i]
            current_label = fetched_labels[i]

            # 计算 L2 距离
            l2_dist = torch.norm(first_vector - current_vector, p=2).item()
            distances_l2.append(l2_dist)

            # 计算 Cosine Similarity (需要增加 batch 维度)
            cos_sim = F.cosine_similarity(first_vector.unsqueeze(0), current_vector.unsqueeze(0)).item()
            cosine_similarities.append(cos_sim)

            print(f"  - Index {current_index} (Label: {current_label}): L2 Dist = {l2_dist:.4f}, Cosine Sim = {cos_sim:.4f}")

            # 同时检查精确相等的情况
            if l2_dist < 1e-6: # 用一个小的阈值代替 torch.equal
                print("    -> VERY CLOSE or IDENTICAL vector found!")

        # --- 分析差异幅度 ---
        print("\n--- Analysis of Differences (Compared to First Vector) ---")
        if distances_l2: # 确保列表不为空
            min_l2 = np.min(distances_l2)
            max_l2 = np.max(distances_l2)
            avg_l2 = np.mean(distances_l2)
            min_cos_sim = np.min(cosine_similarities)
            max_cos_sim = np.max(cosine_similarities)
            avg_cos_sim = np.mean(cosine_similarities)

            print(f"L2 Distances: Min={min_l2:.4f}, Max={max_l2:.4f}, Avg={avg_l2:.4f}")
            print(f"Cosine Similarities: Min={min_cos_sim:.4f}, Max={max_cos_sim:.4f}, Avg={avg_cos_sim:.4f}")

            # --- 初步解读 ---
            # (这些阈值是启发式的，可能需要根据你的具体 embedding 调整)
            l2_low_threshold = 0.1  # 如果最大 L2 距离都小于这个值，可能太近了
            cos_sim_high_threshold = 0.98 # 如果最小相似度都大于这个值，可能太相似了

            if max_l2 < l2_low_threshold and min_cos_sim > cos_sim_high_threshold:
                print("\nWarning: All sampled vectors are VERY CLOSE (low L2 dist, high cos sim) to the first vector.")
                print("Consider if the dataset variation is sufficient for learning.")
            elif avg_cos_sim > 0.95: # 平均相似度非常高
                print("\nNote: Average cosine similarity is very high (> 0.95). Vectors are highly aligned.")
            elif avg_l2 < 1.0: # 平均距离较小 (这个值非常依赖 embedding 的尺度)
                 print("\nNote: Average L2 distance is relatively small.")
            else:
                print("\nObservation: Significant differences observed based on L2 distance and Cosine Similarity.")

            # --- 类内/类间距离分析 (如果抽样中有不同标签) ---
            unique_labels = set(fetched_labels)
            if len(unique_labels) > 1:
                print("\n--- Comparing Within-Class vs Between-Class Differences (Sampled Data) ---")
                same_label_dists_l2 = []
                diff_label_dists_l2 = []
                same_label_sims_cos = []
                diff_label_sims_cos = []

                # 计算所有对之间的距离/相似度
                for i in range(len(fetched_vectors)):
                    for j in range(i + 1, len(fetched_vectors)):
                        vec_i = fetched_vectors[i]
                        vec_j = fetched_vectors[j]
                        label_i = fetched_labels[i]
                        label_j = fetched_labels[j]

                        dist = torch.norm(vec_i - vec_j, p=2).item()
                        sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()

                        if label_i == label_j:
                            same_label_dists_l2.append(dist)
                            same_label_sims_cos.append(sim)
                        else:
                            diff_label_dists_l2.append(dist)
                            diff_label_sims_cos.append(sim)

                # 打印平均值
                if same_label_dists_l2: print(f"  Avg L2 Dist (Same Label Pairs): {np.mean(same_label_dists_l2):.4f}")
                else: print("  No same-label pairs found in sample.")
                if diff_label_dists_l2: print(f"  Avg L2 Dist (Diff Label Pairs): {np.mean(diff_label_dists_l2):.4f}")
                else: print("  No different-label pairs found in sample.")

                if same_label_sims_cos: print(f"  Avg Cos Sim (Same Label Pairs): {np.mean(same_label_sims_cos):.4f}")
                else: print("  No same-label pairs found in sample.")
                if diff_label_sims_cos: print(f"  Avg Cos Sim (Diff Label Pairs): {np.mean(diff_label_sims_cos):.4f}")
                else: print("  No different-label pairs found in sample.")

                # 检查类间距离是否明显大于类内距离 (或者类间相似度是否明显小于类内相似度)
                separation_l2_ok = False
                separation_cos_ok = False
                if diff_label_dists_l2 and same_label_dists_l2 and np.mean(diff_label_dists_l2) > np.mean(same_label_dists_l2) * 1.1: # 类间距离 > 1.1*类内距离
                     separation_l2_ok = True
                if diff_label_sims_cos and same_label_sims_cos and np.mean(diff_label_sims_cos) < np.mean(same_label_sims_cos) * 0.95: # 类间相似度 < 0.95*类内相似度
                     separation_cos_ok = True

                if separation_l2_ok or separation_cos_ok:
                     print("  Observation: Sample suggests some separation between classes (diff-label pairs are further apart / less similar than same-label pairs).")
                elif diff_label_dists_l2 and same_label_dists_l2: # 如果能计算但分界不明显
                     print("  Warning: Separation between classes in the sample is NOT CLEAR. Average distance/similarity for different labels is too close to that of same labels.")
                     print("  This could make it hard for the model to learn.")

            else:
                print("\nNote: All fetched samples have the same label, cannot compare between-class differences.")

        else:
            print("\nNot enough vectors were compared to analyze differences.")