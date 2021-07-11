#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

# Tips: This file is executed for the training of VGG model (and save it).

import os
import argparse

from utils.data_loader import load_dataset
import tensorflow as tf
from utils.VGG_class import VGG

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default="32")  # Max batch size
parser.add_argument("--learning_rate", default="3e-4")  # Learning rate
parser.add_argument("--gpu_device", default="MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0")  # GPU Device
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

# GPU list
gpus = tf.config.experimental.list_physical_devices('GPU')
# ---------------------- GPU显存按需申请 ------------------------
# # 设置按需申请
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# --------------------- GPU显存按比例申请 ------------------------
# 对需要进行限制的GPU进行设置
# for gpu in gpus:
#     tf.config.experimental.set_virtual_device_configuration(gpu, [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

if __name__ == '__main__':
    # Time list
    time_list = []
    # Load dataset
    train_X, train_Y = load_dataset()
    # List for latency
    latency_list = []
    # List for throughput
    throughput_list = []

    batch_size = int(args.batch_size)
    # Model fit
    print("")
    print("############################################################")
    print("#                  Begin model fitting                     #")
    print("############################################################")
    print("")
    print("Device name: %s" % args.gpu_device)
    print("")

    print("Batch size: %d | Learning rate: %f" % (batch_size, float(args.learning_rate)))
    # Model
    cnn_model = VGG(learning_rate=float(args.learning_rate), batch_size=batch_size)
    cnn_model.model.fit(x=train_X, y=train_Y, batch_size=batch_size)

    # Save model
    cnn_model.model.save("./model/vgg_model.h5")

    # Restore model
    # new_model = keras.models.load_model("./model/vgg_model.h5")
    # new_model.summary()
