#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import numpy as np
import tensorflow as tf
import pandas as pd


## Data Loading ##
def load_data(filefolder):
    filefolder = './data/' + filefolder
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label


# Load dataset
def load_dataset():
    # Dataset
    train_X, train_Y = load_data('train')  # 训练集，[8169，73, 398]
    # Dataset reshape
    input_shape = (train_X[0].shape[0], train_X[0].shape[1], 1)
    reshaped_train_X = list(train_X)
    label_2D = tf.one_hot(train_Y, 2)

    for i in range(0, len(reshaped_train_X), 1):
        # Transfer int8 to int32
        reshaped_train_X[i] = reshaped_train_X[i].astype(np.float32)
        reshaped_train_X[i] = tf.reshape(reshaped_train_X[i], shape=input_shape)
    reshaped_train_X = np.array(reshaped_train_X)

    return reshaped_train_X, label_2D
