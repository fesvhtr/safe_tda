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
    """
    使用 Persistence Landscapes 将持久性区间向量化。

    Args:
        persistence_intervals: GUDHI persistence() 返回的列表 [(dim, (birth, death)), ...]。
        num_landscapes: 要计算的景观函数数量 (k)。
        resolution: 在 [x_min, x_max] 区间内采样景观函数的点数。
        x_min: 采样区间的起始点。
        x_max: 采样区间的结束点。

    Returns:
        np.ndarray: 拼接后的景观函数采样值向量 (长度为 num_landscapes * resolution)，如果输入无效则返回空数组。
    """
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


def vectorize_persistence_image(persistence_intervals,
                                bandwidth: float = 0.1,
                                weight_func=lambda x: x[1] - x[0],  # Weight by persistence
                                resolution: list = [20, 20],
                                im_range=None):  # [birth_min, death_min, birth_max, death_max]
    """
    使用 Persistence Images 将持久性区间向量化。

    Args:
        persistence_intervals: GUDHI persistence() 返回的列表。
        bandwidth: 高斯核的带宽。
        weight_func: 应用于每个 (birth, death) 点的权重函数。默认按持久性加权。
        resolution: 输出图像的分辨率 [width_pixels, height_pixels]。
        im_range: 图像覆盖的范围 [b_min, d_min, b_max, d_max]。如果为 None，通常由库根据数据计算。

    Returns:
        np.ndarray: 展平后的持久性图像向量 (长度为 resolution[0] * resolution[1])，如果输入无效则返回空数组。
    """
    # 同样需要按维度分组
    diagrams_by_dim = {}
    for dim, (birth, death) in persistence_intervals:
        # 通常只对有限的点进行图像化，无穷远点意义不同
        if not np.isinf(death):
            if dim not in diagrams_by_dim:
                diagrams_by_dim[dim] = []
            diagrams_by_dim[dim].append([birth, death])

    # 选择维度 1 (H1) 作为示例
    target_dim = 1
    if target_dim in diagrams_by_dim and diagrams_by_dim[target_dim]:
        diagram_h1 = np.array(diagrams_by_dim[target_dim])

        # 如果 im_range 未指定，从数据中估算一个合理的范围
        auto_im_range = im_range
        if auto_im_range is None and diagram_h1.shape[0] > 0:
            b_min, b_max = diagram_h1[:, 0].min(), diagram_h1[:, 0].max()
            d_min, d_max = diagram_h1[:, 1].min(), diagram_h1[:, 1].max()
            # 稍微扩展范围以包含所有点
            padding_b = (b_max - b_min) * 0.05 if (b_max > b_min) else 0.1
            padding_d = (d_max - d_min) * 0.05 if (d_max > d_min) else 0.1
            auto_im_range = [b_min - padding_b, d_min - padding_d,
                             b_max + padding_b, d_max + padding_d]
            # 确保下限不为负，并且死亡 > 诞生
            auto_im_range[0] = max(0, auto_im_range[0])
            auto_im_range[1] = max(0, auto_im_range[1])
            if auto_im_range[2] <= auto_im_range[0]: auto_im_range[2] = auto_im_range[0] + 0.1
            if auto_im_range[3] <= auto_im_range[1]: auto_im_range[3] = auto_im_range[1] + 0.1
            # 确保死亡总是大于等于诞生
            if auto_im_range[3] < auto_im_range[0]: auto_im_range[3] = auto_im_range[
                                                                           0] + 0.1  # Adjust death max if needed
            if auto_im_range[1] > auto_im_range[2]: auto_im_range[1] = auto_im_range[
                                                                           2] - 0.1  # Adjust death min if needed

        if auto_im_range is None:  # Still None if diagram was empty
            warnings.warn(
                f"Cannot compute Persistence Image for dim {target_dim}: diagram is empty or im_range not specified.")
            return np.zeros(resolution[0] * resolution[1])

        PI = gd.representations.PersistenceImage(bandwidth=bandwidth, weight=weight_func,
                                                 resolution=resolution, im_range=auto_im_range)

        # fit_transform 需要一个包含 diagrams 的列表
        image_vectors = PI.fit_transform([diagram_h1])
        # 返回第一个 diagram 的图像向量
        return image_vectors[0]
    else:
        warnings.warn(f"No valid finite persistence intervals found for dimension {target_dim} for Persistence Image.")
        return np.zeros(resolution[0] * resolution[1])


def vectorize_betti_curve(betti_data, num_samples=50, sample_scales=None):
    """
    通过在特定尺度上采样 Betti 曲线来将其向量化。

    Args:
        betti_data (np.ndarray): Betti 数数组 (num_thresholds x num_dimensions)，由 compute_betti_curves 返回。
        num_samples (int): 如果 sample_scales 未提供，则在 [0, max_scale] 范围内均匀采样多少个点。
        sample_scales (np.ndarray, optional): 指定要采样的确切尺度值。如果提供，则忽略 num_samples。

    Returns:
        np.ndarray: 拼接后的 Betti 数向量 (长度为 num_dimensions * len(sampled_scales))。
    """
    if betti_data is None or betti_data.size == 0:
        return np.array([])

    num_thresholds, num_dims = betti_data.shape

    # 假设原始阈值均匀分布，用于插值
    # 注意：这依赖于 compute_betti_curves 使用了均匀的 thresholds
    # 如果需要更精确，应该将原始 thresholds 也传入此函数
    original_max_scale = 1.0  # 假设最大尺度为1，需要根据实际情况调整或传入
    # TODO: Ideally, pass the actual thresholds used to generate betti_data
    original_thresholds = np.linspace(0, original_max_scale, num_thresholds)

    if sample_scales is None:
        # 在原始尺度范围内均匀采样 num_samples 个点
        sampled_scales = np.linspace(original_thresholds.min(), original_thresholds.max(), num_samples)
    else:
        # 使用用户提供的尺度
        sampled_scales = np.array(sample_scales)

    vectorized_betti = np.zeros((len(sampled_scales), num_dims))

    for dim in range(num_dims):
        # 使用线性插值在采样点上获取 Betti 值
        vectorized_betti[:, dim] = np.interp(sampled_scales, original_thresholds, betti_data[:, dim])

    # 展平成一维向量
    return vectorized_betti.flatten()


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