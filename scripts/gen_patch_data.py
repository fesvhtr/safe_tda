import random
import json

def generate_random_groups(total=5000, group_size=50, num_groups=250):
    groups = []
    for _ in range(num_groups):
        group = random.sample(range(total), group_size)
        groups.append(group)
    return groups

# 生成随机组
random_groups = generate_random_groups()

# 保存为 JSON 文件
output_path = r"H:\ProjectsPro\safe_tda\data\dataset\test_patch_id.json"
with open(output_path, "w") as f:
    json.dump(random_groups, f, indent=2)

print(f"Saved {len(random_groups)} groups to {output_path}")
