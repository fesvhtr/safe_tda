# Safe TDA
Final Project for KU ECCE 794 Topological Data Analysis

### Quick Start
```
git clone https://github.com/fesvhtr/safe_tda.git
pip install ghudi
```
Then Download data from [huggingface](https://huggingface.co/datasets/fesvhtr/safe_tda)  
Put images in data/datasets, Put cache in data/cache
If run from begin rather than use cached TDA features, please use the [Visu](https://huggingface.co/datasets/aimagelab/ViSU-Text) Dataset 

### Run Training
Use ```patch_train.py``` and ```patch_reg_train.py``` to train MLP
replace TDA feature path use the cache path you download