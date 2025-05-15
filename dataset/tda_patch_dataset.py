import os
import json
import atexit
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import joblib
from filelock import FileLock       # pip install filelock
from tqdm import tqdm
from tda.compute import analyze_and_plot_tda
from tda.vectorize import (
    vectorize_landscape,
    vectorize_persistence_image,
    vectorize_betti_curve,
    vectorize_simple_stats,
)

class TDAPatchClsDataset(Dataset):
    def __init__(
        self,
        nsfw_embeddings,
        nsfw_group_indices_path,
        safe_embeddings,
        safe_group_indices_path,
        tda_method="landscape",
        cache_path=None,
        plot=False,
        force_recompute=False,
        dump_every=250, # 写盘间隔
        return_mode="first"              
    ):
        # -------- 初始化参数 --------
        self.tda_method = [tda_method] if isinstance(tda_method, str) else tda_method
        self.cache_path = cache_path
        self.plot = plot
        self.force_recompute = force_recompute
        self.dump_every = dump_every
        self._info_printed = True
        self.return_mode = return_mode

        # -------- 并发写盘相关 --------
        self._cache = {}
        self._dirty = False          # 有新增向量
        self._lock = FileLock(f"{cache_path}.lock") if cache_path else None
        atexit.register(lambda: self._maybe_dump_cache(force=True))

        # -------- 读取已存在的缓存 --------
        if cache_path and os.path.exists(cache_path):
            try:
                self._cache = joblib.load(cache_path)
                print(f"[TDA] Loaded cache from {cache_path}, total groups: {len(self._cache)}")
            except Exception as e:
                print(f"[TDA] Failed to load cache: {e}")

        # -------- 读取分组索引 --------
        with open(nsfw_group_indices_path) as f:
            nsfw_groups = json.load(f)
        with open(safe_group_indices_path) as f:
            safe_groups = json.load(f)

        # -------- 构建数据索引 --------
        self.group_data = []
        for i, idx in enumerate(nsfw_groups):
            self.group_data.append({"embedding": nsfw_embeddings,
                                    "indices": idx, "label": 1,
                                    "key": f"group_nsfw_{i}"})
        for i, idx in enumerate(safe_groups):
            self.group_data.append({"embedding": safe_embeddings,
                                    "indices": idx, "label": 0,
                                    "key": f"group_safe_{i}"})

        self.num_nsfw = len(nsfw_groups)
        self.num_safe = len(safe_groups)
        self.total_groups = len(self.group_data)

        # -------- 缓存统计 --------
        self.cache_stats = {
            "total_cached_groups": len(self._cache),
            "cached_vec_count": 0,
            "method_counts": defaultdict(int)
        }
        for v in self._cache.values():
            for m in self.tda_method:
                if f"vec_{m}" in v:
                    self.cache_stats["cached_vec_count"] += 1
                    self.cache_stats["method_counts"][m] += 1

    # ------------------------------------------------------------
    # 公共工具：按需写缓存（并发安全）
    # ------------------------------------------------------------
    def _maybe_dump_cache(self, force=False):
        if not self.cache_path or (not force and not self._dirty):
            return
        with self._lock:
            joblib.dump(self._cache, self.cache_path, compress=3)
        self._dirty = False
        if force:
            print(f"[TDA] Cache saved to {self.cache_path}")

    # ------------------------------------------------------------
    def __len__(self):
        # 检查是否已经打印过信息
        if not self._info_printed:
            minfo = ", ".join(f"{m}:{self.cache_stats['method_counts'].get(m,0)}"
                                for m in self.tda_method)
            print(f"[TDA Dataset] Total groups: {self.total_groups} "
                    f"(NSFW:{self.num_nsfw}, Safe:{self.num_safe})")
            print(f"[TDA Dataset] Cache: {self.cache_stats['total_cached_groups']} groups, "
                    f"{self.cache_stats['cached_vec_count']} vecs [{minfo}]")

            # 设置标志位，表示信息已打印
            self._info_printed = True

        return self.total_groups

    # ------------------------------------------------------------
    def _compute_and_cache(self, group_dict):
        emb = group_dict["embedding"][group_dict["indices"]]   # already on CPU
        gkey = group_dict["key"]
        gcache = self._cache.get(gkey, {})

        persistence = betti = thresholds = None # 先初始化为 None
        run_base_tda = False
        if self.force_recompute:
            run_base_tda = True
        elif not gcache:
            run_base_tda = True
        elif any(f"vec_{m}" not in gcache for m in self.tda_method):
          run_base_tda = True
        if run_base_tda:
            persistence, betti, thresholds = analyze_and_plot_tda(
                emb, label=None, complex='rips',
                max_filt_scale=2.0, max_hom_dim=3,
                threshold_count=100, plot=self.plot
            )


        # -------- 生成所需向量 --------
        for method in self.tda_method:
            mkey = f"vec_{method}"
            if not self.force_recompute and mkey in gcache:
                continue
            if method == "landscape":
                vec = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
            elif method == "image":
                vec = vectorize_persistence_image(persistence, target_dim=1, bandwidth = 0.4, resolution=[20, 20])
                # NOTE: log1p 归一化
                vec = np.log1p(vec)
            elif method == "betti":
                vec = vectorize_betti_curve(
                    betti_data=betti,
                    thresholds=thresholds,
                    num_samples=50,
                    max_hom_dim=2  # 假设我们关心 H0、H1、H2
                )
            elif method == "stats":
                vec = vectorize_simple_stats(persistence, max_dim=1)
            else:
                raise ValueError(f"Invalid tda_method: {method}")
            gcache[mkey] = vec

        # -------- 更新缓存 & 写盘标记 --------
        self._cache[gkey] = gcache
        self._dirty = True
        if len(self._cache) % self.dump_every == 0:
            self._maybe_dump_cache()

        return gcache[f"vec_{self.tda_method[0]}"]


    def __getitem__(self, idx):
        gdict = self.group_data[idx]
        gkey = gdict["key"]

        if (self.force_recompute or
            gkey not in self._cache or
            any(f"vec_{m}" not in self._cache.get(gkey, {}) for m in self.tda_method)): # 使用 .get 避免 gkey 不存在时出错
            # print(f"[TDA] Computing TDA for {gkey} ({idx}/{self.total_groups})")
            self._compute_and_cache(gdict)

        if self.return_mode == "concat":
            vecs_to_concat = []
            for method_name in self.tda_method:
                current_vec = self._cache[gkey][f"vec_{method_name}"]
                if isinstance(current_vec, np.ndarray):
                    vecs_to_concat.append(current_vec.flatten())
                else:
                    vecs_to_concat.append(np.array(current_vec).flatten())
            vec = np.concatenate(vecs_to_concat)
        elif self.return_mode == "first":
            primary_key = f"vec_{self.tda_method[0]}"
            vec = self._cache[gkey][primary_key]
        else:
            raise ValueError(f"Invalid return_mode: {self.return_mode}. Must be 'first' or 'concat'.")

        return torch.tensor(vec, dtype=torch.float32), torch.tensor(gdict["label"], dtype=torch.long)

    # ------------------------------------------------------------
    def __del__(self):
        # 确保对象销毁时落盘
        try:
            self._maybe_dump_cache(force=True)
        except Exception:
            pass


class TDAPatchRegDataset(Dataset):
    def __init__(
        self,
        nsfw_embeddings,
        safe_embeddings,
        mix_group_indices_path,
        tda_method="landscape",
        cache_path=None,
        plot=False,
        force_recompute=False,
        dump_every=250, # 写盘间隔
        return_mode="first"              
    ):
        # -------- 初始化参数 --------
        self.tda_method = [tda_method] if isinstance(tda_method, str) else tda_method
        self.cache_path = cache_path
        self.plot = plot
        self.force_recompute = force_recompute
        self.dump_every = dump_every
        self._info_printed = True
        self.return_mode = return_mode
        self.nsfw_embeddings = nsfw_embeddings
        self.safe_embeddings = safe_embeddings

        # -------- 并发写盘相关 --------
        self._cache = {}
        self._dirty = False          # 有新增向量
        self._lock = FileLock(f"{cache_path}.lock") if cache_path else None
        atexit.register(lambda: self._maybe_dump_cache(force=True))

        # -------- 读取已存在的缓存 --------
        if cache_path and os.path.exists(cache_path):
            try:
                self._cache = joblib.load(cache_path)
                print(f"[TDA] Loaded cache from {cache_path}, total groups: {len(self._cache)}")
            except Exception as e:
                print(f"[TDA] Failed to load cache: {e}")

        # -------- 读取分组索引 --------
        with open(mix_group_indices_path) as f:
            mix_groups = json.load(f)

        # -------- 构建数据索引 --------
        self.group_data = []
        for i, idx in enumerate(mix_groups):
            safe_cnt = len(idx["safe"])
            nsfw_cnt = len(idx["nsfw"])
            # print(f"safe: {safe_cnt}, nsfw: {nsfw_cnt}")
            # NOTE: 这里的比例是 nsfw / (safe + nsfw)
            propotion = float(nsfw_cnt / (safe_cnt + nsfw_cnt))
            # print(f"propotion: {propotion}")
            self.group_data.append({"safe_indices": idx["safe"], "nsfw_indices" :idx["nsfw"], "label": propotion,
                                    "key": f"group_mix_{i}"})

        self.total_groups = len(self.group_data)

        # -------- 缓存统计 --------
        self.cache_stats = {
            "total_cached_groups": len(self._cache),
            "cached_vec_count": 0,
            "method_counts": defaultdict(int)
        }
        for v in self._cache.values():
            for m in self.tda_method:
                if f"vec_{m}" in v:
                    self.cache_stats["cached_vec_count"] += 1
                    self.cache_stats["method_counts"][m] += 1

    # ------------------------------------------------------------
    # 公共工具：按需写缓存（并发安全）
    # ------------------------------------------------------------
    def _maybe_dump_cache(self, force=False):
        if not self.cache_path or (not force and not self._dirty):
            return
        with self._lock:
            joblib.dump(self._cache, self.cache_path, compress=3)
        self._dirty = False
        if force:
            print(f"[TDA] Cache saved to {self.cache_path}")

    # ------------------------------------------------------------
    def __len__(self):
        # 检查是否已经打印过信息
        if not self._info_printed:
            minfo = ", ".join(f"{m}:{self.cache_stats['method_counts'].get(m,0)}"
                                for m in self.tda_method)
            print(f"[TDA Dataset] Total groups: {self.total_groups}")
            print(f"[TDA Dataset] Cache: {self.cache_stats['total_cached_groups']} groups, "
                    f"{self.cache_stats['cached_vec_count']} vecs [{minfo}]")

            # 设置标志位，表示信息已打印
            self._info_printed = True

        return self.total_groups

    # ------------------------------------------------------------
    def _compute_and_cache(self, group_dict):
        safe_emb = self.safe_embeddings[group_dict["safe_indices"]]  # already on CPU
        nsfw_emb = self.nsfw_embeddings[group_dict["nsfw_indices"]]
        emb = np.concatenate([safe_emb, nsfw_emb], axis=0)
        gkey = group_dict["key"]
        gcache = self._cache.get(gkey, {})

        persistence = betti = thresholds = None # 先初始化为 None
        run_base_tda = False
        if self.force_recompute:
            run_base_tda = True
        elif not gcache:
            run_base_tda = True
        elif any(f"vec_{m}" not in gcache for m in self.tda_method):
          run_base_tda = True
        if run_base_tda:
            persistence, betti, thresholds = analyze_and_plot_tda(
                emb, label=None, complex='rips',
                max_filt_scale=2.0, max_hom_dim=3,
                threshold_count=100, plot=self.plot
            )

        # -------- 生成所需向量 --------
        for method in self.tda_method:
            mkey = f"vec_{method}"
            if not self.force_recompute and mkey in gcache:
                continue
            if method == "landscape":
                vec = vectorize_landscape(persistence, num_landscapes=3, resolution=100)
            elif method == "image":
                vec = vectorize_persistence_image(persistence, target_dim=1, bandwidth = 0.4, resolution=[20, 20])
                # NOTE: log1p 归一化
                vec = np.log1p(vec)
            elif method == "betti":
                vec = vectorize_betti_curve(
                    betti_data=betti,
                    thresholds=thresholds,
                    num_samples=50,
                    max_hom_dim=2  # 假设我们关心 H0、H1、H2
                )
            elif method == "stats":
                vec = vectorize_simple_stats(persistence, max_dim=1)
            else:
                raise ValueError(f"Invalid tda_method: {method}")
            gcache[mkey] = vec

        # -------- 更新缓存 & 写盘标记 --------
        self._cache[gkey] = gcache
        self._dirty = True
        if len(self._cache) % self.dump_every == 0:
            self._maybe_dump_cache()

        return gcache[f"vec_{self.tda_method[0]}"]


    def __getitem__(self, idx):
        gdict = self.group_data[idx]
        gkey = gdict["key"]

        if (self.force_recompute or
            gkey not in self._cache or
            any(f"vec_{m}" not in self._cache.get(gkey, {}) for m in self.tda_method)): # 使用 .get 避免 gkey 不存在时出错
            # print(f"[TDA] Computing TDA for {gkey} ({idx}/{self.total_groups})")
            self._compute_and_cache(gdict)

        if self.return_mode == "concat":
            vecs_to_concat = []
            for method_name in self.tda_method:
                current_vec = self._cache[gkey][f"vec_{method_name}"]
                if isinstance(current_vec, np.ndarray):
                    vecs_to_concat.append(current_vec.flatten())
                else:
                    vecs_to_concat.append(np.array(current_vec).flatten())
            vec = np.concatenate(vecs_to_concat)
        elif self.return_mode == "first":
            primary_key = f"vec_{self.tda_method[0]}"
            vec = self._cache[gkey][primary_key]
        else:
            raise ValueError(f"Invalid return_mode: {self.return_mode}. Must be 'first' or 'concat'.")

        return torch.tensor(vec, dtype=torch.float32), torch.tensor(gdict["label"], dtype=torch.float32)

    # ------------------------------------------------------------
    def __del__(self):
        # 确保对象销毁时落盘
        try:
            self._maybe_dump_cache(force=True)
        except Exception:
            pass