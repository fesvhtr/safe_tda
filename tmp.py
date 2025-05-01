import json

with open(r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train.json", "r") as f:
    data = json.load(f)

for item_ in data:
    item_["coco_id"] = str(item_["coco_id"]).zfill(12)
data5k = data[:5000]
with open(r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train_5K.json", "w") as f:
    json.dump(data5k, f)
with open(r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train.json", "w") as f:
    json.dump(data[5000:], f)