import random
import json

def generate_random_groups(total, group_size, num_groups):
    groups = []
    for _ in range(num_groups):
        group = random.sample(range(total), group_size)
        groups.append(group)
    return groups

# 生成随机组
total = 20000
group_size = 75
num_groups = 5000
random_groups = generate_random_groups(total, group_size, num_groups)

# 保存为 JSON 文件
output_path = rf"H:\ProjectsPro\safe_tda\data\dataset\patch_ids\train_patch_id_ss{group_size}g{num_groups}.json"
with open(output_path, "w") as f:
    json.dump(random_groups, f, indent=2)

print(f"Saved {len(random_groups)} groups to {output_path}")
