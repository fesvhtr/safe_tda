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
        self.tda_method = tda_method
        self.cache_path = cache_path
        self.plot = plot
        self.force_recompute = force_recompute
        self._cache = {}

        if self.cache_path and os.path.exists(self.cache_path):
            self._cache = torch.load(self.cache_path)
            print(f"[TDA] Loaded cache from {self.cache_path}, total groups: {len(self._cache)}")

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
            "total_cached_groups": 0,
            "cached_vec_count": 0
        }
        if self._cache:
            method_key = f"vec_{self.tda_method}"
            self.cache_stats["total_cached_groups"] = len(self._cache)
            self.cache_stats["cached_vec_count"] = sum(
                1 for v in self._cache.values() if method_key in v
            )

    def __len__(self):
        if not hasattr(self, "_printed_len_info"):
            print(f"[TDA Dataset] Total groups: {self.total_groups} "
                  f"(NSFW: {self.num_nsfw}, Safe: {self.num_safe})")
            print(f"[TDA Dataset] Cache status: {self.cache_stats['total_cached_groups']} group entries, "
                  f"{self.cache_stats['cached_vec_count']} with method '{self.tda_method}'")
            self._printed_len_info = True
        return self.total_groups

    def _compute_and_cache(self, group_dict):
        group_embed = group_dict["embedding"][group_dict["indices"]]
        group_key = group_dict["key"]

        # Compute TDA
        persistence, betti = analyze_and_plot_tda(
            group_embed,
            label=None,
            complex='rips',
            max_filt_scale=2.0,
            max_hom_dim=3,
            threshold_count=100,
            plot=self.plot,
        )

        # Vectorize
        method_key = f"vec_{self.tda_method}"
        if self.tda_method == "landscape":
            vec = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
        elif self.tda_method == "image":
            vec = vectorize_persistence_image(persistence, resolution=[20, 20])
        elif self.tda_method == "betti":
            vec = vectorize_betti_curve(betti, num_samples=50,
                                        sample_scales=np.linspace(0.0, 2.0, 50))
        elif self.tda_method == "stats":
            vec = vectorize_simple_stats(persistence, max_dim=1)
        elif self.tda_method == "concat":
            vec1 = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
            vec2 = vectorize_persistence_image(persistence, resolution=[20, 20])
            vec3 = vectorize_betti_curve(betti, num_samples=50,
                                         sample_scales=np.linspace(0.0, 2.0, 50))
            vec4 = vectorize_simple_stats(persistence, max_dim=1)
            vec = np.concatenate([vec1, vec2, vec3, vec4], axis=-1)
        else:
            raise ValueError("Invalid tda_method")

        # 写入缓存（覆盖或新增）
        self._cache[group_key] = {
            "persistence": persistence,
            "betti": betti,
            method_key: vec
        }

        if self.cache_path:
            torch.save(self._cache, self.cache_path)

        return vec

    def __getitem__(self, idx):
        group_dict = self.group_data[idx]
        group_key = group_dict["key"]
        method_key = f"vec_{self.tda_method}"

        if (not self.force_recompute and
                group_key in self._cache and
                method_key in self._cache[group_key]):
            vec = self._cache[group_key][method_key]
        else:
            vec = self._compute_and_cache(group_dict)

        vec_tensor = torch.tensor(vec, dtype=torch.float32)
        label = torch.tensor(group_dict["label"], dtype=torch.long)
        return vec_tensor, label
