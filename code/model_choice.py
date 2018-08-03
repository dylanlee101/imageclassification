# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 15:24
# @Author  : Dylan
# @File    : model_choice.py
# @Email   : wenyili@buaa.edu.cn
from config import config
from model.resnet import ResnetBuilder

def model_c():
    if config.model_choice == "resnet18":
        model = ResnetBuilder.build_resnet_18((config.channels, config.norm_size, config.norm_size), config.num_classes)
    elif config.model_choice == "resnet34":
        model = ResnetBuilder.build_resnet_34((config.channels, config.norm_size, config.norm_size), config.num_classes)
    elif config.model_choice == "resnet50":
        model = ResnetBuilder.build_resnet_50((config.channels, config.norm_size, config.norm_size), config.num_classes)
    elif config.model_choice == "resnet101":
        model = ResnetBuilder.build_resnet_101((config.channels, config.norm_size, config.norm_size), config.num_classes)
    elif config.model_choice == "resnet152":
        model = ResnetBuilder.build_resnet_152((config.channels, config.norm_size, config.norm_size), config.num_classes)
    elif config.model_choice == "xception":
        model = ""
    elif config.model_choice == "inception":
        model = ""
    elif config.model_choice == "incetption_resnet_v2":
        model = ""
    elif config.model_choice == "vgg16":
        model = ""
    else:
        print("Your model not found")
    return model