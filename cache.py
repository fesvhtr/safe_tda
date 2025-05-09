import os
import torch
# import numpy as np # TDAPatchDataset handles numpy internally if needed
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- 让脚本能找到你的模块 ---
# 假设你的项目结构类似，并且此脚本放在项目根目录或能找到 utils, dataset, model 的地方
# 如果你的脚本位置不同，可能需要调整 sys.path
# import sys
# sys.path.append('/path/to/your/project/root') # 如果需要的话取消注释并修改路径

try:
    from utils.dataset_utils import load_embeddings
    from utils.model_utils import load_clip
    from dataset.tda_patch_dataset import TDAPatchDataset
except ImportError as e:
    print(f"错误：无法导入项目模块: {e}")
    print("请确保脚本运行在正确的目录，或者你的项目路径在 PYTHONPATH 中。")
    exit()
# --- ---


# --- 1. 在这里硬编码你的参数 ---
MODALITY = "text"  # 或 "image"
TDA_METHOD = ["landscape", "image", "betti", "stats"] # 必须是列表
MODEL_NAME = "ViT-L/14"  # 或 "longclip"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 基础项目路径 (如果你的数据/代码结构不同，请修改)
BASE_PROJECT_PATH = "/home/muzammal/Projects/safe_proj/safe_tda" # !! 修改为你的实际路径 !!

# 索引和缓存文件的路径 (基于 BASE_PROJECT_PATH 和 MODALITY 构建)
TRAIN_NSFW_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/train_patch_id_ns75g5000.json")
TRAIN_SAFE_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/train_patch_id_ss75g5000.json")
TRAIN_CACHE_PATH   = os.path.join(BASE_PROJECT_PATH, f"data/cache/{MODALITY}_patch_train.pkl") # 使用 .pkl 而不是 .joblib

VAL_NSFW_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/val_patch_id_ns75g500.json")
VAL_SAFE_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/val_patch_id_ss75g500.json")
VAL_CACHE_PATH   = os.path.join(BASE_PROJECT_PATH, f"data/cache/{MODALITY}_patch_val.pkl") # 使用 .pkl

TEST_NSFW_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/test_patch_id_ns75g500.json")
TEST_SAFE_INDICES = os.path.join(BASE_PROJECT_PATH, "data/dataset/patch_ids/test_patch_id_ss75g500.json")
TEST_CACHE_PATH   = os.path.join(BASE_PROJECT_PATH, f"data/cache/{MODALITY}_patch_test.pkl") # 使用 .pkl

# 设置此缓存脚本的 force_recompute 行为
# False: 仅计算缓存中缺失的项目 (适合续传或添加新方法)
# True:  重新计算所有内容，确保缓存最新 (耗时更长)
FORCE_RECOMPUTE_CACHE = False # !! 根据需要调整 !!
# --- 参数结束 ---


def cache_dataset(dataset_name, dataset_instance):
    num_items = len(dataset_instance)

    # 创建 DataLoader
    loader = DataLoader(
        dataset=dataset_instance,
        batch_size=1,          # 确保 __getitem__ 被独立调用
        shuffle=False,
        pin_memory=False       # 通常不需要
    )

    # 迭代 DataLoader 以触发缓存
    for _ in tqdm(loader):
        pass # 主要工作在后台 worker 的 __getitem__ 调用中完成

    print(f"--- 完成缓存 {dataset_name} ---")


if __name__ == "__main__":
    print("--- 开始为训练、验证、测试集进行 TDA 预计算 ---")
    print(f"使用设备: {DEVICE}")
    print(f"CLIP 模型: {MODEL_NAME}")
    print(f"模态: {MODALITY}")
    print(f"TDA 方法: {TDA_METHOD}")
    print(f"本脚本强制重算设置: {FORCE_RECOMPUTE_CACHE}")

    # --- 0. 加载 CLIP 模型 (load_embeddings 需要) ---
    print("正在加载 CLIP 模型...")
    clip_model, clip_preprocess, clip_tokenizer = load_clip(MODEL_NAME, DEVICE)
    print("-" * 30)

    # --- 1. 缓存训练集 ---
    print("正在加载 训练集 嵌入向量...")
    safe_emb_train, nsfw_emb_train, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer, DEVICE, split="train", modality=MODALITY)
    # train_set = TDAPatchDataset(
    #     nsfw_embeddings=nsfw_emb_train,
    #     nsfw_group_indices_path=TRAIN_NSFW_INDICES,
    #     safe_embeddings=safe_emb_train,
    #     safe_group_indices_path=TRAIN_SAFE_INDICES,
    #     tda_method=TDA_METHOD,
    #     cache_path=TRAIN_CACHE_PATH, # 使用 .pkl
    #     plot=False,
    #     force_recompute=FORCE_RECOMPUTE_CACHE # 使用脚本顶部的设置
    # )

    train_set = TDAPatchDataset(
        nsfw_embeddings=nsfw_emb_train,
        nsfw_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/train_patch_id_ns50-100g5000.json",  
        safe_embeddings=safe_emb_train,
        safe_group_indices_path="/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids/train_patch_id_ss50-100g5000.json",
        tda_method=TDA_METHOD,
        cache_path=f"/home/muzammal/Projects/safe_proj/safe_tda/data/cache/{MODALITY}_patch_train_hy.pkl",
        plot=False,
        force_recompute=FORCE_RECOMPUTE_CACHE
    )
    cache_dataset("训练集", train_set)
    # 清理内存 (如果嵌入向量很大)
    del safe_emb_train, nsfw_emb_train
    if 'cuda' in DEVICE: torch.cuda.empty_cache()

    # # --- 2. 缓存验证集 ---
    # print("\n正在加载 验证集 嵌入向量...")
    # safe_emb_val, nsfw_emb_val, _ = load_embeddings(clip_model, clip_preprocess,clip_tokenizer, DEVICE, split="val", modality=MODALITY)
    # val_set = TDAPatchDataset(
    #     nsfw_embeddings=nsfw_emb_val,
    #     nsfw_group_indices_path=VAL_NSFW_INDICES,
    #     safe_embeddings=safe_emb_val,
    #     safe_group_indices_path=VAL_SAFE_INDICES,
    #     tda_method=TDA_METHOD,
    #     cache_path=VAL_CACHE_PATH, # 使用 .pkl
    #     plot=False,
    #     force_recompute=FORCE_RECOMPUTE_CACHE # 使用脚本顶部的设置
    # )
    # cache_dataset("验证集", val_set)
    # del safe_emb_val, nsfw_emb_val
    # if 'cuda' in DEVICE: torch.cuda.empty_cache()

    # --- 3. 缓存测试集 ---
    # print("\n正在加载 测试集 嵌入向量...")
    # safe_emb_test, nsfw_emb_test, _ = load_embeddings(clip_model, clip_preprocess, clip_tokenizer, DEVICE, split="test", modality=MODALITY)
    # test_set = TDAPatchDataset(
    #     nsfw_embeddings=nsfw_emb_test,
    #     nsfw_group_indices_path=TEST_NSFW_INDICES,
    #     safe_embeddings=safe_emb_test,
    #     safe_group_indices_path=TEST_SAFE_INDICES,
    #     tda_method=TDA_METHOD,
    #     cache_path=TEST_CACHE_PATH, # 使用 .pkl
    #     plot=False,
    #     force_recompute=FORCE_RECOMPUTE_CACHE # 使用脚本顶部的设置
    # )
    # cache_dataset("测试集", test_set)
    # print("dataset len:", len(test_set))
    # print("\n--- 所有缓存任务已完成 ---")