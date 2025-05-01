import os
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import time
from PIL import Image
from io import BytesIO
import json

def download_image(args):
    """下载单个图片的函数"""
    url, save_path = args
    try:
        # 如果文件已经存在，跳过下载
        if os.path.exists(save_path):
            return f"Already exists: {save_path}"

        # 发送请求并下载图片
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # 验证图片能否正确打开
            try:
                img = Image.open(BytesIO(response.content))
                img.verify()  # 验证图片完整性

                # 保存图片
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return f"Success: {save_path}"
            except Exception as e:
                return f"Invalid image: {url}, Error: {str(e)}"
        else:
            return f"Failed to download: {url}, Status code: {response.status_code}"
    except Exception as e:
        return f"Error downloading {url}: {str(e)}"


def download_coco_images(output_dir, num_workers=16, max_images=None):
    """下载COCO-Karpathy测试集中的所有图片"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    print("Loading dataset...")
    ds = load_dataset("yerevann/coco-karpathy", split="train")

    # 限制要下载的图片数量（如果指定）
    if max_images:
        ds = ds.select(range(min(max_images, len(ds))))

    print(f"Found {len(ds)} images to download")

    # 准备下载任务
    download_tasks = []
    for item in ds:
        url = item['url']
        if not url.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 从URL中提取文件名
        filename = os.path.basename(url).split("_")[-1]  # 去掉查询参数
        save_path = os.path.join(output_dir, filename)
        download_tasks.append((url, save_path))

    # 使用线程池并行下载
    print(f"Downloading {len(download_tasks)} images to {output_dir}...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks)))

    # 统计下载结果
    success = sum(1 for r in results if r.startswith("Success"))
    already = sum(1 for r in results if r.startswith("Already"))
    failed = len(results) - success - already

    print(f"Download complete: {success} new downloads, {already} already existed, {failed} failed")

def download_by_id(json_file, save_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    download_tasks = []
    for item_ in tqdm(data):
        coco_id = item_["coco_id"]
        url_prefix = "http://images.cocodataset.org/val2014/COCO_val2014_"
        url_suffix = ".jpg"
        # fill coco_id with 12 digits with leading zeros
        # coco_id = str(coco_id).zfill(12)
        url = url_prefix + coco_id + url_suffix
        # create save path
        save_path = os.path.join(save_dir, f"{coco_id}.jpg")
        if os.path.exists(save_path):
            continue
        # download image
        download_tasks.append((url, save_path))
    # 使用线程池并行下载
    print(f"Downloading {len(download_tasks)} images to {save_dir}...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks)))


if __name__ == "__main__":
    # 设置输出目录
    output_dir = r"H:\ProjectsPro\safe_tda\data\dataset\val_coco"
    json_file = r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train_5K.json"
    # 启动下载，可以设置max_images参数来限制下载数量
    # 例如: download_coco_images(output_dir, max_images=100)
    # download_coco_images(output_dir)
    download_by_id(json_file, output_dir)