import random
import json
import os
from collections import defaultdict
from tqdm import tqdm

def generate_random_groups(total, group_size, num_groups):
    groups = []
    for _ in range(num_groups):
        group = random.sample(range(total), group_size)
        groups.append(group)
    return groups
def generate_variable_size_groups(total, min_group_size, max_group_size, num_groups):
    """
    生成具有可变大小的随机组。
    每个组的大小在 [min_group_size, max_group_size] 之间随机选择。
    """
    groups = []
    for _ in range(num_groups):
        # 为当前组随机选择一个大小
        current_group_size = random.randint(min_group_size, max_group_size)
        # 从总数中随机采样指定大小的组，确保不重复
        group = random.sample(range(total), current_group_size)
        groups.append(group)
    return groups

# --- 参数设置 ---
total_items = 5000  # 总的可选项目数量
min_n_size = 50     # 组大小的最小值 (N_min)
max_n_size = 100    # 组大小的最大值 (N_max)
num_groups_to_generate = 1000 # 要生成的组的数量

# 生成随机组
print(f"正在生成 {num_groups_to_generate} 个随机组...")
print(f"每个组的大小将在 [{min_n_size}, {max_n_size}] 之间随机选择。")
random_groups_variable_size = generate_variable_size_groups(
    total_items,
    min_n_size,
    max_n_size,
    num_groups_to_generate
)

# --- 保存为 JSON 文件 ---
# 修改文件名以反映可变大小 N
# 你可以根据需要调整文件名的格式，例如使用 "Nvar" 或范围 "N50-100"
output_directory = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"创建目录: {output_directory}")

# 文件名可以这样表示范围，或者你选择其他方式
output_filename = f"test_patch_id_ss{min_n_size}-{max_n_size}g{num_groups_to_generate}.json"
output_path = os.path.join(output_directory, output_filename)

with open(output_path, "w") as f:
    json.dump(random_groups_variable_size, f, indent=2)

print(f"已保存 {len(random_groups_variable_size)} 个组到 {output_path}")


print("\n前5个组的大小示例:")
for i, group in enumerate(random_groups_variable_size[:5]):
    print(f"组 {i+1}: 大小 = {len(group)}")





# # 生成随机组
# total = 20000
# group_size = 75
# num_groups = 5000
# random_groups = generate_random_groups(total, group_size, num_groups)

# # 保存为 JSON 文件
# output_path = rf"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\train_patch_id_ss{group_size}g{num_groups}.json"
# with open(output_path, "w") as f:
#     json.dump(random_groups, f, indent=2)

# print(f"Saved {len(random_groups)} groups to {output_path}")
