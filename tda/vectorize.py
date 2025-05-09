# -*- coding: utf-8 -*-
# Imports (确保你已经安装了 gudhi)
import gudhi as gd
import gudhi.representations  # 显式导入 GUDHI 的表示模块
import numpy as np
import warnings


# --- 向量化方法函数 ---

def vectorize_landscape(persistence_intervals,
                        num_landscapes: int = 3,
                        resolution: int = 100,
                        x_min: float = 0.0,
                        x_max: float = 1.0):
    # 过滤掉无穷大的死亡时间点，因为 Landscape 通常处理有限区间
    finite_intervals = [pair for pair in persistence_intervals if not np.isinf(pair[1][1])]
    if not finite_intervals:
        warnings.warn("No finite persistence intervals found for Landscape vectorization.")
        # 返回一个期望长度的全零向量
        return np.zeros(num_landscapes * resolution)

    # GUDHI Landscape 类需要一个包含 diagrams 的列表作为输入
    # 并且每个 diagram 是一个 (N, 2) 的 NumPy 数组，只包含 [birth, death] 对
    # 我们需要按维度分组
    diagrams_by_dim = {}
    for dim, (birth, death) in finite_intervals:
        if dim not in diagrams_by_dim:
            diagrams_by_dim[dim] = []
        diagrams_by_dim[dim].append([birth, death])

    # 将每个维度的 diagram 转为 numpy array
    diagram_list_for_gudhi = []
    max_dim_present = -1
    if diagrams_by_dim:
        max_dim_present = max(diagrams_by_dim.keys())
        # 确保所有维度都有，即使是空的，以保持一致性（尽管 Landscape 通常分别处理维度）
        for d in range(max_dim_present + 1):
            if d in diagrams_by_dim:
                diagram_list_for_gudhi.append(np.array(diagrams_by_dim[d]))
            else:
                diagram_list_for_gudhi.append(np.empty((0, 2)))  # 空 diagram
    else:
        # 如果完全没有有限区间，创建一个空 diagram 列表
        diagram_list_for_gudhi = [np.empty((0, 2))]

    # --- 注意：GUDHI Landscape 通常是对单个维度的 diagram 操作 ---
    # --- 这里我们为每个维度计算景观并拼接，或选择一个维度 ---
    # --- 选择维度 1 (H1) 作为示例，通常信息量更丰富 ---
    target_dim = 1
    if target_dim < len(diagram_list_for_gudhi) and diagram_list_for_gudhi[target_dim].shape[0] > 0:
        LS = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=resolution,
                                          sample_range=[x_min, x_max])
        # fit_transform 需要一个包含 diagrams 的列表
        landscape_vectors = LS.fit_transform([diagram_list_for_gudhi[target_dim]])
        # 返回第一个（也是唯一一个）diagram 的景观向量
        return landscape_vectors[0]
    else:
        warnings.warn(f"No valid persistence intervals found for dimension {target_dim} for Landscape vectorization.")
        # 返回一个期望长度的全零向量
        return np.zeros(num_landscapes * resolution)


def vectorize_persistence_image(
        persistence_intervals,
        target_dim: int = 1,
        bandwidth: float = 0.3,
        resolution: tuple = (20, 20),
        im_range: list = None,
        normalize: bool = False
    ) -> np.ndarray:
    # 1. 收集 finite interval
    diagrams_by_dim = {}
    for dim, (b, d) in persistence_intervals:
        if not np.isinf(d):
            diagrams_by_dim.setdefault(dim, []).append([b, d])

    # 2. 如果没有这一维，返回全零
    if target_dim not in diagrams_by_dim or len(diagrams_by_dim[target_dim]) == 0:
        warnings.warn(f"No finite intervals for dimension {target_dim}")
        return np.zeros(resolution[0] * resolution[1])

    # 3. 转成 array
    diagram = np.array(diagrams_by_dim[target_dim])  # shape (N,2)
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]

    # 4. 自动估算 im_range
    if im_range is None:
        b_min, b_max = births.min(), births.max()
        p_min, p_max = persistences.min(), persistences.max()
        pad_b = (b_max - b_min)*0.05 if b_max > b_min else 0.1
        pad_p = (p_max - p_min)*0.05 if p_max > p_min else 0.1
        im_range = [
            max(0.0, b_min - pad_b),
            max(0.0, p_min - pad_p),
            b_max + pad_b,
            p_max + pad_p
        ]

    # 5. 构造 PersistenceImage —— 不传 weight，使用默认 w(b,d)=d−b
    PI = gd.representations.PersistenceImage(
        bandwidth=bandwidth,
        resolution=resolution,
        im_range=im_range
    )

    # 6. fit_transform 需要一个 list 输入
    image_vec = PI.fit_transform([diagram])[0]  # shape = (W*H,)

    # 7. 归一化（可选）
    if normalize:
        m = image_vec.max()
        if m > 0:
            image_vec = image_vec / m

    return image_vec


import numpy as np

def vectorize_betti_curve(
        betti_data: np.ndarray,
        thresholds: np.ndarray,
        num_samples: int = 50,
        sample_scales: np.ndarray = None,
        max_hom_dim: int = 3
    ) -> np.ndarray:
    # 1. 参数检查
    if betti_data is None or betti_data.size == 0:
        return np.zeros((max_hom_dim + 1) * (sample_scales.size if sample_scales is not None else num_samples))

    T, D_actual = betti_data.shape
    D_expected = max_hom_dim + 1

    # 2. 用 0 填充至统一同调维度数
    full = np.zeros((T, D_expected), dtype=betti_data.dtype)
    full[:, :D_actual] = betti_data

    # 3. 确定采样尺度
    if sample_scales is None:
        sampled_scales = np.linspace(thresholds.min(), thresholds.max(), num_samples)
    else:
        sampled_scales = np.array(sample_scales)
    # 裁剪到真实阈值范围，防止外插
    sampled_scales = np.clip(sampled_scales, thresholds.min(), thresholds.max())

    # 4. 对每个维度做插值
    vec = np.zeros((sampled_scales.size, D_expected), dtype=betti_data.dtype)
    for dim in range(D_expected):
        vec[:, dim] = np.interp(sampled_scales, thresholds, full[:, dim])

    # 5. 扁平化并返回
    return vec.flatten()



def vectorize_simple_stats(persistence_intervals, max_dim=1):
    """
    计算持久性区间的一些简单统计量作为特征向量。

    Args:
        persistence_intervals: GUDHI persistence() 返回的列表。
        max_dim: 计算统计量的最高维度。

    Returns:
        np.ndarray: 包含统计特征的向量。
    """
    features = []
    for dim in range(max_dim + 1):
        # 提取当前维度的有限持久性对
        pairs = np.array([[b, d] for d_int, (b, d) in persistence_intervals if d_int == dim and not np.isinf(d)])

        # 特征 1: 当前维度有限持久性点的数量
        num_points = pairs.shape[0]
        features.append(num_points)

        if num_points > 0:
            persistence_values = pairs[:, 1] - pairs[:, 0]
            birth_values = pairs[:, 0]
            death_values = pairs[:, 1]

            # 特征 2: 总持久性
            features.append(np.sum(persistence_values))
            # 特征 3: 平均持久性
            features.append(np.mean(persistence_values))
            # 特征 4: 持久性方差
            features.append(np.var(persistence_values))
            # 特征 5: 平均诞生时间
            features.append(np.mean(birth_values))
            # 特征 6: 平均死亡时间
            features.append(np.mean(death_values))
            # 特征 7: 最大持久性
            features.append(np.max(persistence_values))
        else:
            # 如果没有点，添加 0 作为统计特征
            features.extend([0.0] * 6)  # 对应总、均、方差、均b、均d、最大持久性

    # 检查是否有无限持久性点 (通常 H0 有一个)
    inf_points_count = [0] * (max_dim + 1)
    for dim, (birth, death) in persistence_intervals:
        if np.isinf(death) and dim <= max_dim:
            inf_points_count[dim] += 1
    # 特征 8 onwards: 每个维度的无限持久性点数量
    features.extend(inf_points_count)

    return np.array(features)