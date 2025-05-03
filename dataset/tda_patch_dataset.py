import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tda.compute import analyze_and_plot_tda
from tda.vectorize import (
    vectorize_landscape,
    vectorize_persistence_image,
    vectorize_betti_curve,
    vectorize_simple_stats,
)
import json
from collections import defaultdict
import joblib
class TDAPatchDataset(Dataset):
    def __init__(self,
                 nsfw_embeddings,
                 nsfw_group_indices_path,
                 safe_embeddings,
                 safe_group_indices_path,
                 tda_method="landscape",
                 cache_path=None,
                 plot=False,
                 force_recompute=False):
        if isinstance(tda_method, str):
            self.tda_method = [tda_method]
        else:
            self.tda_method = tda_method
        self.cache_path = cache_path
        self.plot = plot
        self.force_recompute = force_recompute
        self._cache = {}

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                self._cache = joblib.load(self.cache_path)
                print(f"[TDA] Loaded cache from {self.cache_path}, total groups: {len(self._cache)}")
            except Exception as e:
                print(f"[TDA] Failed to load cache: {e}")
                self._cache = {}

        with open(nsfw_group_indices_path, "r") as f:
            nsfw_groups = json.load(f)
        with open(safe_group_indices_path, "r") as f:
            safe_groups = json.load(f)

        self.group_data = []
        for i, indices in enumerate(nsfw_groups):
            self.group_data.append({
                "embedding": nsfw_embeddings,
                "indices": indices,
                "label": 1,
                "key": f"group_nsfw_{i}"
            })
        for i, indices in enumerate(safe_groups):
            self.group_data.append({
                "embedding": safe_embeddings,
                "indices": indices,
                "label": 0,
                "key": f"group_safe_{i}"
            })
        self.num_nsfw = len(nsfw_groups)
        self.num_safe = len(safe_groups)
        self.total_groups = len(self.group_data)

        self.cache_stats = {
            "total_cached_groups": len(self._cache),
            "cached_vec_count": 0,
            "method_counts": defaultdict(int)
        }

        for v in self._cache.values():
            for method in self.tda_method:
                key = f"vec_{method}"
                if key in v and v[key] is not None:
                    self.cache_stats["cached_vec_count"] += 1
                    self.cache_stats["method_counts"][method] += 1

    def __len__(self):
        if not hasattr(self, "_printed_len_info"):
            # 构建 method-wise 缓存统计信息字符串
            methods_info = ', '.join(
                f"{m}: {self.cache_stats['method_counts'].get(m, 0)}"
                for m in self.tda_method
            )

            print(f"[TDA Dataset] Total groups: {self.total_groups} "
                  f"(NSFW: {self.num_nsfw}, Safe: {self.num_safe})")
            print(f"[TDA Dataset] Cache status: {self.cache_stats['total_cached_groups']} group entries, "
                  f"{self.cache_stats['cached_vec_count']} cached vectors [{methods_info}]")
            self._printed_len_info = True
        return self.total_groups

    def _compute_and_cache(self, group_dict):
        group_embed = group_dict["embedding"][group_dict["indices"]]
        group_key = group_dict["key"]

        group_cache = self._cache.get(group_key, {})

        # 尝试加载或重新计算 persistence 和 betti
        if self.force_recompute or "persistence" not in group_cache or "betti" not in group_cache:
            persistence, betti = analyze_and_plot_tda(
                group_embed,
                label=None,
                complex='rips',
                max_filt_scale=2.0,
                max_hom_dim=3,
                threshold_count=100,
                plot=self.plot,
            )
            group_cache["persistence"] = persistence
            group_cache["betti"] = betti
        else:
            persistence = group_cache["persistence"]
            betti = group_cache["betti"]

        # 增量更新每种方法的向量（只计算缺失项）
        for method in self.tda_method:
            method_key = f"vec_{method}"
            if not self.force_recompute and method_key in group_cache:
                continue

            if method == "landscape":
                vec = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
            elif method == "image":
                vec = vectorize_persistence_image(persistence, resolution=[20, 20])
            elif method == "betti":
                vec = vectorize_betti_curve(betti, num_samples=50,
                                            sample_scales=np.linspace(0.0, 2.0, 50))
            elif method == "stats":
                vec = vectorize_simple_stats(persistence, max_dim=1)
            elif method == "concat":
                vec1 = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
                vec2 = vectorize_persistence_image(persistence, resolution=[20, 20])
                vec3 = vectorize_betti_curve(betti, num_samples=50,
                                             sample_scales=np.linspace(0.0, 2.0, 50))
                vec4 = vectorize_simple_stats(persistence, max_dim=1)
                vec = np.concatenate([vec1, vec2, vec3, vec4], axis=-1)
            else:
                raise ValueError(f"Invalid tda_method: {method}")

            group_cache[method_key] = vec

        # 保存更新后的 group_cache
        self._cache[group_key] = group_cache

        if self.cache_path:
            joblib.dump(self._cache, self.cache_path)

        return group_cache[f"vec_{self.tda_method[0]}"]

    def __getitem__(self, idx):
        group_dict = self.group_data[idx]
        group_key = group_dict["key"]
        primary_method = self.tda_method[0]
        method_key = f"vec_{primary_method}"

        # 如果任一方法没缓存，或者强制重算，则触发计算
        if (self.force_recompute or
                group_key not in self._cache or
                any(f"vec_{method}" not in self._cache[group_key] for method in self.tda_method)):
            _ = self._compute_and_cache(group_dict)

        # 仅返回第一个 method 的向量用于训练
        vec = self._cache[group_key][method_key]
        vec_tensor = torch.tensor(vec, dtype=torch.float32)
        label = torch.tensor(group_dict["label"], dtype=torch.long)
        return vec_tensor, label


