# -*- coding: utf-8 -*-
# Imports (确保你已经安装了 gudhi 和 numpy)
import gudhi as gd
import gudhi.wasserstein  # 需要导入 wasserstein 模块
import numpy as np
import warnings


# --- 辅助函数：从 persistence 列表中提取指定维度的 diagram ---
def _extract_diagram(persistence_intervals, dimension: int):
    """
    从 GUDHI persistence() 的输出中提取指定维度的有限点，
    并格式化为 GUDHI 距离函数所需的 NumPy 数组。
    """
    # 提取有限点 (birth, death) 对
    diagram = np.array([[birth, death]
                        for dim, (birth, death) in persistence_intervals
                        if dim == dimension and not np.isinf(death)])

    # 如果没有点，返回一个 shape 为 (0, 2) 的空数组
    if diagram.shape[0] == 0:
        return np.empty((0, 2))
    return diagram


# --- 距离计算函数 ---

def compute_bottleneck_distance(persistence1, persistence2, dimension: int = 1, e: float = None):
    """
    计算两个持久性区间列表之间在指定维度上的瓶颈距离。

    Args:
        persistence1: 第一个 GUDHI persistence() 返回的列表。
        persistence2: 第二个 GUDHI persistence() 返回的列表。
        dimension: 要比较的同调维度 (例如 0 代表 H0, 1 代表 H1)。
        e (float, optional): 误差容忍度。如果为 0，使用精确但可能较慢的算法。
                             如果不为 0 (或 None，默认为最小正 double)，
                             使用通常快得多的近似算法。默认为 None (使用近似)。

    Returns:
        float: 计算出的瓶颈距离。如果某个 diagram 在该维度为空，可能返回 0 或 inf，需注意。
               返回 -1.0 表示计算出错或输入为空。
    """
    diag1 = _extract_diagram(persistence1, dimension)
    diag2 = _extract_diagram(persistence2, dimension)

    # 如果两个 diagram 都为空，距离为 0
    if diag1.shape[0] == 0 and diag2.shape[0] == 0:
        return 0.0
    # 如果只有一个 diagram 为空，理论上距离是无限大，但 GUDHI 实现可能不同。
    # GUDHI 的 bottleneck_distance 可以处理这种情况（将空 diagram 视为只有对角线）。

    try:
        # 注意：GUDHI 的 bottleneck_distance 函数需要 diagram 作为输入
        # GUDHI v3.5.0 之后推荐使用 gudhi.bottleneck.bottleneck_distance
        # 早期版本是 gudhi.bottleneck_distance
        # 我们假设使用较新版本（如果出错，可能需要调整导入或函数名）
        if hasattr(gd, 'bottleneck') and hasattr(gd.bottleneck, 'bottleneck_distance'):
            distance = gd.bottleneck.bottleneck_distance(diag1, diag2, e=e)
        else:
            # 尝试旧版接口
            distance = gd.bottleneck_distance(diag1, diag2, e=e)
        return distance
    except Exception as err:
        warnings.warn(f"Error computing Bottleneck distance for dim {dimension}: {err}")
        return -1.0  # 返回错误指示


def compute_wasserstein_distance(persistence1, persistence2, dimension: int = 1, order: float = 1.0,
                                 internal_p: float = np.inf):
    """
    计算两个持久性区间列表之间在指定维度上的 Wasserstein 距离。

    Args:
        persistence1: 第一个 GUDHI persistence() 返回的列表。
        persistence2: 第二个 GUDHI persistence() 返回的列表。
        dimension: 要比较的同调维度。
        order (float): Wasserstein 距离的阶数 q (>= 1)。常用 1 或 2。
        internal_p (float): 计算点对距离时使用的 L^p 范数 (>= 1)。常用 2 (欧氏距离) 或 np.inf (L_infinity, 即棋盘距离)。

    Returns:
        float: 计算出的 Wasserstein 距离。返回 -1.0 表示计算出错或输入为空。
    """
    diag1 = _extract_diagram(persistence1, dimension)
    diag2 = _extract_diagram(persistence2, dimension)

    # 如果两个 diagram 都为空，距离为 0
    if diag1.shape[0] == 0 and diag2.shape[0] == 0:
        return 0.0
    # Wasserstein 距离函数通常也能处理一个为空的情况

    try:
        # GUDHI 的 Wasserstein 距离函数位于 gudhi.wasserstein 模块
        distance = gd.wasserstein.wasserstein_distance(diag1, diag2, order=order, internal_p=internal_p)
        return distance
    except Exception as err:
        warnings.warn(f"Error computing Wasserstein distance for dim {dimension}: {err}")
        return -1.0  # 返回错误指示