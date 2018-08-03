# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 13:11
# @Author  : Dylan
# @File    : utils.py
# @Email   : wenyili@buaa.edu.cn
import os
import cv2
import numpy as np
from model_choice import model_c
# import matplotlib.pyplot as plt
from config import config
from keras.preprocessing.image import img_to_array
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras.utils import to_categorical
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def path_list(path):
    path = path
    data_folder_list1 = list(map(lambda x: path + x, os.listdir(path)))
    if path == config.train_path:
        # no val path
        if os.path.exists(config.other_train_path):
            path2 = config.other_train_path
            data_folder_list2 = list(map(lambda x: path2 + x, os.listdir(path2)))
            data_folder_list = data_folder_list1 + data_folder_list2
        else:
            data_folder_list = data_folder_list1
    else:
        data_folder_list = data_folder_list1
    return data_folder_list

def path_choice(val = False):
    if config.model_train:
        if os.path.exists(config.val_path) and val:
            path = config.val_path
            data_folder_list = path_list(path)
        else:
            path = config.train_path
            data_folder_list = path_list(path)
    else:
        path = config.test_path
        data_folder_list = path_list(path)
    return data_folder_list



def load_data(val = False):
    data_folder_list = path_choice(val = val)
    images = []
    labels = []
    file_name = []
    for files in tqdm(data_folder_list):
        files_list = os.listdir(files)
        for file in tqdm(files_list):
            name = files + "/" + file
            #             print(name.split("/")[-2])
            if name[-3:] == "jpg":
                label = name.split("/")[-2]
                labels.append(label)
                file_name.append(name)
                #                 print(name)
                image = cv2.imread(name)
                image1 = cv2.resize(image, (config.norm_size, config.norm_size))
                image = img_to_array(image1)
                images.append(image)
            else:
                continue

    images = np.array(images, dtype="float") / 255.0
    print(images.shape)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], 1)
    labels = to_categorical(labels, num_classes=config.num_classes)
    return images, labels


def model_train(x_train, y_train, x_test, y_test):

    lr_reduce = ReduceLROnPlateau(factor=config.lr, monitor='val_acc', patience=5, verbose=1)
    early_stop = EarlyStopping(monitor='val_acc', patience=21, verbose=1)
    csv_log = CSVLogger(config.log_path)
    checkpoint = ModelCheckpoint(config.checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
    model = model_c()

    if config.load_weigth:
        if os.path.exists(config.checkpoint_path):
            print("\nLoading best weights.\n")
            model.load_weights(config.checkpoint_path)
        elif os.path.exists(config.save_model_path):
            print("\nLoading my weights.\n")
            model.load_weights(config.save_model_path, by_name=True)
        else:
            pass

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])

    if not config.aug:
        print("Not using data augmentation.")
        H = model.fit(x_train, y_train,
                      batch_size=config.train_batch_size,
                      nb_epoch=config.max_epoch,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      callbacks=[lr_reduce, early_stop, csv_log, checkpoint])
    else:
        print("Using real-time data augmentation.")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        datagen.fit(x_train)

        H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=config.train_batch_size),
                                steps_per_epoch=x_train.shape[0] // config.train_batch_size,
                                validation_data=(x_test, y_test),
                                epochs=config.max_epoch, verbose=1, max_q_size=100,
                                callbacks=[lr_reduce, early_stop, csv_log, checkpoint])
    print("model training is finished...")
    model.save(config.save_model_path)

    # plt.style.use("ggplot")
    # plt.figure()
    # N = config.max_epoch
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    # plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    # plt.title("Training Loss and Accuracy on Audio Test classifier")
    # plt.xlabel("Epoch ")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig(config.plot_path)

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def equal_hist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst


def h(image):
    size = (image.shape[1], image.shape[0])
    iLR = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    h = image.shape[1]
    w = image.shape[0]
    for i in range(h):
        for j in range(w):
            # iUD[h-1-i,j] = image[i,j]
            iLR[i, w - 1 - j] = image[i, j]
            # iAcross[h-1-i,w-1-j] = image[i,j]
    gray = cv2.cvtColor(iLR, cv2.COLOR_BGR2GRAY)
    iLR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return iLR