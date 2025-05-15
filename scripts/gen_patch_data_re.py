import random
import json
import os

def generate_id_mixture_final_v2( 
    output_filepath: str,
    num_total_batches: int,
    fixed_batch_size: int = None,
    batch_size_range: tuple = (50, 100),
    global_id_population: range = range(20000)
):
    """
    为一个数据集分割生成包含随机数量 nsfw/safe ID 的批次数据。
    可以选择固定批次大小或随机批次大小。
    """
    print(f"开始为 {output_filepath} 生成数据 (最终ID池逻辑)...")
    all_batches_data = []

    # 检查 batch_size_range 的有效性
    if fixed_batch_size is None:
        if batch_size_range[0] < 0 or batch_size_range[1] < batch_size_range[0]:
            raise ValueError("batch_size_range 参数无效，应为 (min_size, max_size) 格式且 min_size <= max_size。")
    else:
        if fixed_batch_size <= 0:
            raise ValueError("fixed_batch_size 必须大于 0。")
        print(f"已选择固定批次大小: {fixed_batch_size}")

    for i in range(num_total_batches):
        # 选择批次大小
        if fixed_batch_size is not None:
            batch_total_size_N = fixed_batch_size
        else:
            batch_total_size_N = random.randint(batch_size_range[0], batch_size_range[1])

        # 随机确定 NSFW ID 的数量
        num_nsfw = random.randint(0, batch_total_size_N)
        num_safe = batch_total_size_N - num_nsfw

        sampled_nsfw_ids = random.sample(global_id_population, min(num_nsfw, len(global_id_population)))
        sampled_safe_ids = random.sample(global_id_population, min(num_safe, len(global_id_population)))

        batch_entry = {
            "safe": sorted(sampled_safe_ids),
            "nsfw": sorted(sampled_nsfw_ids)
        }
        all_batches_data.append(batch_entry)

        if (i + 1) % 500 == 0:
            print(f"  已生成 {i + 1} / {num_total_batches} 个批次...")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")

    # 保存为 JSON 文件
    with open(output_filepath, "w") as f:
        json.dump(all_batches_data, f, indent=2)

    print(f"成功！已将 {len(all_batches_data)} 个批次的数据保存到 {output_filepath}")
    print("-" * 30)


if __name__ == "__main__":
    # --- 全局配置 ---
    TOTAL_UNIQUE_ITEMS = 5000
    global_item_ids_pool = range(TOTAL_UNIQUE_ITEMS) 

    base_output_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids" 

    # 固定批次大小示例
    # generate_id_mixture_final_v2( 
    #     output_filepath=os.path.join(base_output_dir, "test_patch_id_mix75g1000.json.json"),
    #     num_total_batches=10000,
    #     fixed_batch_size=75,
    #     global_id_population=global_item_ids_pool
    # )

    # 随机批次大小示例
    generate_id_mixture_final_v2( 
        output_filepath=os.path.join(base_output_dir, "test_patch_id_mix50-100g1000.json"),
        num_total_batches=1000,
        batch_size_range=(50, 100),
        global_id_population=global_item_ids_pool
    )

    print("所有 ID 混合数据集文件（最终ID池逻辑）生成完毕。")
