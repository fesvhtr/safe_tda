import json

with open(r"/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_train.json", "r") as f:
    data = json.load(f)

for item_ in data:
    item_["coco_id"] = str(item_["coco_id"]).zfill(12)
data5k = data[:20000]
with open(r"/home/muzammal/Projects/safe_proj/safe_tda/data/dataset/ViSU-Text_train_20K.json", "w") as f:
    json.dump(data5k, f)