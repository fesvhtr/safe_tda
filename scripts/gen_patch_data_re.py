import random
import json
import os

def generate_id_mixture_final( # 函数名再次调整以示区别
    output_filepath: str,
    num_total_batches: int,
    batch_total_size_N: int,
    global_id_population: range # 单个全局ID池
):
    """
    为一个数据集分割生成包含随机数量 nsfw/safe ID 的批次数据。
    NSFW ID 从全局池中独立抽取，确保 NSFW 内部不重复。
    Safe ID 从全局池中独立抽取，确保 Safe 内部不重复。
    一个 ID 理论上可以同时出现在一个批次的 nsfw 和 safe 列表中。

    每个批次是一个字典 {"safe": [ids], "nsfw": [ids]}。
    safe 和 nsfw ID 的总数为 batch_total_size_N。
    实际的 NSFW 比例将在 Dataset 类中计算。
    """
    print(f"开始为 {output_filepath} 生成数据 (最终ID池逻辑)...")
    all_batches_data = []

    # 基本检查，确保ID池至少能满足单次最大可能的抽样需求
    if len(global_id_population) < batch_total_size_N and batch_total_size_N > 0 : # 如果N为0则不适用
         pass # 如果允许N大于池大小，则sample会报错，这里先不强制退出
              # random.sample(population, k) 当 k > len(population) 会抛出 ValueError

    for i in range(num_total_batches):
        # 1. 为当前批次随机确定 NSFW ID 的数量
        num_nsfw = random.randint(0, batch_total_size_N)
        num_safe = batch_total_size_N - num_nsfw

        sampled_nsfw_ids = []
        sampled_safe_ids = []

        # 2. 从全局ID池中为当前批次独立抽取 NSFW ID
        # 确保 global_id_population 至少有 num_nsfw 个元素
        if num_nsfw > 0:
            if len(global_id_population) < num_nsfw:
                print(f"警告：批次 {i+1}，全局ID池大小 ({len(global_id_population)}) 小于请求的 NSFW 数量 ({num_nsfw})。将抽取所有可用ID。")
                sampled_nsfw_ids = random.sample(global_id_population, len(global_id_population))
            else:
                sampled_nsfw_ids = random.sample(global_id_population, num_nsfw)

        # 3. 从全局ID池中为当前批次独立抽取 Safe ID
        # 确保 global_id_population 至少有 num_safe 个元素
        if num_safe > 0:
            if len(global_id_population) < num_safe:
                print(f"警告：批次 {i+1}，全局ID池大小 ({len(global_id_population)}) 小于请求的 Safe 数量 ({num_safe})。将抽取所有可用ID。")
                sampled_safe_ids = random.sample(global_id_population, len(global_id_population))
            else:
                sampled_safe_ids = random.sample(global_id_population, num_safe)
        
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
    BATCH_SIZE_N = 75

    # 现在只有一个全局ID池
    TOTAL_UNIQUE_ITEMS = 20000
    global_item_ids_pool = range(TOTAL_UNIQUE_ITEMS) 

    print(f"全局 ID 池范围: {global_item_ids_pool.start} - {global_item_ids_pool.stop - 1} (共 {len(global_item_ids_pool)} 个ID)")
    print(f"每个批次总大小 N: {BATCH_SIZE_N}")
    print("JSON 将只包含 safe 和 nsfw ID 列表。比例将在 Dataset 类中计算。")
    print("NSFW 和 Safe ID 将从同一个全局池中独立抽取。")
    print("-" * 30)

    base_output_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/patch_ids" 

    test_config = {
        "output_filepath": os.path.join(base_output_dir, f"trainS_patch_id_mix75_g10000.json"),
        "num_total_batches": 1000,
    }
    generate_id_mixture_final( 
        output_filepath=test_config["output_filepath"],
        num_total_batches=test_config["num_total_batches"],
        batch_total_size_N=BATCH_SIZE_N,
        global_id_population=global_item_ids_pool
    )

    print("所有 ID 混合数据集文件（最终ID池逻辑）生成完毕。")
