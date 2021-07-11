#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


# Implementation of VGGNet16
class VGG:
    def __init__(self, learning_rate, batch_size):
        # Learning rate
        self.lr = learning_rate
        # Batch size
        self.batch_size = batch_size
        # CNN net
        self.model = self.build_net()

    def build_net(self):
        # Layers
        model = Sequential()
        # Layer 1
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(MaxPool2D(2, 2))
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(MaxPool2D(2, 2))
        # Layer 3
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(MaxPool2D(2, 2))
        # Layer 4
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(MaxPool2D(2, 2))
        # Layer 5
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu))
        model.add(MaxPool2D(2, 2))
        # Flatten
        model.add(Flatten(input_shape=[-1, 9 * 25 * 512]))
        # FC layer 1
        model.add(Dense(4096))
        model.add(Dropout(0.5))
        # FC layer 2
        model.add(Dense(4096))
        model.add(Dropout(0.5))
        # FC layer 3
        model.add(Dense(2, activation='softmax'))

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
        return model
