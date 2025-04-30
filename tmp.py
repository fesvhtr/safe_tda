import json

with open(r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train.json", "r") as f:
    data = json.load(f)
    data = data[:5000]
with open(r"H:\ProjectsPro\safe_tda\data\dataset\ViSU-Text_train_5K.json", "w") as f:
    json.dump(data, f)