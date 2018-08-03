# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 13:11
# @Author  : Dylan
# @File    : config.py
# @Email   : wenyili@buaa.edu.cn
class Config():
    model = ''

    train_path = ''
    other_train_path = ''
    val_path = ''
    test_path = ""
    checkpoint_path = '../save_models/best_model.h5'
    save_model_path = '../save_models/my_model.h5'
    save_sub_path = '../outputs/result.csv'
    log_path ="../analyse/log.csv"
    plot_path = '../analyse/plot.png'

    load_model_path = None
    load_weigth = False
    model_train = True
    norm_size = 224
    num_classes = 4
    channels = 3
    train_batch_size = 20
    val_batch_size = 20



    use_gpu = "6"
    aug = True
    ratio = 0.2
    num_workers = 2
    print_freq = 8

    max_epoch = 200
    lr = 1e-4
    lr_reduce = 0.5
    weight_reduce = 0.5

    model_choice = 'resnet18'

config = Config()