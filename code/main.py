# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 13:11
# @Author  : Dylan
# @File    : main.py
# @Email   : wenyili@buaa.edu.cn
import os
import numpy as np
from utils import load_data,model_train
from sklearn.model_selection import train_test_split
from config import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.use_gpu
np.random.seed(2018)

def _main():
    if config.model_train:
        if not os.path.exists(config.val_path):
            x, y = load_data()
            print(len(x), len(y))
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=config.ratio)
        else:
            x_train,y_train = load_data()
            x_val,y_val = load_data(val = True)
        model_train(x_train, y_train, x_val, y_val)
    else:
        x,y = load_data()

if __name__ == '__main__':
    _main()
