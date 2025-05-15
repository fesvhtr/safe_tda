import os
import shutil

def merge_folders(src_dir1, src_dir2, dest_dir):
    # 如果目标文件夹不存在，创建它
    os.makedirs(dest_dir, exist_ok=True)

    # 处理第一个源文件夹
    for filename in os.listdir(src_dir1):
        src_path = os.path.join(src_dir1, filename)
        dest_path = os.path.join(dest_dir, filename)

        # 避免覆盖，处理重名文件
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1

        shutil.move(src_path, dest_path)

    # 处理第二个源文件夹
    for filename in os.listdir(src_dir2):
        src_path = os.path.join(src_dir2, filename)
        dest_path = os.path.join(dest_dir, filename)

        # 避免覆盖，处理重名文件
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                counter += 1

        shutil.move(src_path, dest_path)

    print(f"✅ 合并完成，所有文件已移动到 {dest_dir}")

# —— 使用示例 ——
src_dir1 = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored_5-20k"
src_dir2 = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored_5k"
dest_dir = "/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/train_FLUX_Unsensored_20k"

merge_folders(src_dir1, src_dir2, dest_dir)
