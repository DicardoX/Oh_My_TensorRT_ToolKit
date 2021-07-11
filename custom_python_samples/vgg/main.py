#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import time
import numpy as np
import argparse
import keras2onnx
import onnx
import tf2onnx.convert

from utils.data_loader import load_dataset
import tensorflow as tf


# Tips: This file is executed for the predicting of VGG model with different batch_size and GPU instance.
#       First load the pre-trained model.

parser = argparse.ArgumentParser()
parser.add_argument("--min_batch_size", default="1")  # Min batch size
parser.add_argument("--max_batch_size", default="128")  # Max batch size
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
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

if __name__ == '__main__':
    # Load dataset
    train_X, train_Y = load_dataset()
    # List for latency
    latency_list = []
    # List for throughput
    throughput_list = []

    batch_size = int(args.min_batch_size)
    # Model fit
    print("")
    print("############################################################")
    print("#                Begin model predicting                    #")
    print("############################################################")
    print("")
    print("Device name: %s" % args.gpu_device)
    print("")

    print("Loading pre-trained VGG model...")
    vgg_model = tf.keras.models.load_model("./model/vgg_model.h5")
    # cnn_model.summary()

    # --------------------- Convert to onnx ----------------------- #
    # Version 1 (keras2onnx): AttributeError: 'KerasTensor' object has no attribute 'graph
    # onnx_vgg_model = keras2onnx.convert_keras(vgg_model, vgg_model.name, target_opset=10, channel_first_inputs='input_1')
    # Version 2 (tf2onnx)
    spec = (tf.TensorSpec((None, 73, 398, 1), tf.float32, name="input"),)
    onnx_vgg_model, _ = tf2onnx.convert.from_keras(vgg_model, input_signature=spec)
    # Save onnx model
    onnx_model_path = "./model/vgg_model.onnx"
    onnx.save_model(onnx_vgg_model, onnx_model_path)
    # ------------------------------------------------------------- #

    counter = 0
    while batch_size <= int(args.max_batch_size):
        # Time list
        time_list = []

        counter += 1
        print("")
        print("Iteration %d | Batch size: %d | Learning rate: %f" % (counter, batch_size, float(args.learning_rate)))
        idx = 0
        while idx + batch_size < int(len(train_X)):
            time_mark = time.time()
            prediction = vgg_model.predict(train_X[idx:(idx+batch_size)])
            time_list.append((time.time() - time_mark) * 1000)
            idx += batch_size
        # Remove the first elm, too big
        time_list.pop(0)
        print("")
        print(len(time_list))
        print("")
        # Latency, remove the cost in CPU
        latency = round(float(np.mean(time_list)), 3) - 30
        latency_list.append(latency)
        # Throughput
        throughput = int((1000 / latency) * batch_size)
        throughput_list.append(throughput)
        print("")
        print("Model predict completed! | Batch size: %d | Average latency per batch: %.3f ms | Throughput: %d" % (
            batch_size, latency, throughput))
        print("")
        batch_size *= 2

    print("")
    # Model fit
    print("############################################################")
    print("#                         RESULT                           #")
    print("############################################################")
    print("")
    print("Device name: %s" % args.gpu_device)
    print("")

    for i in range(len(latency_list)):
        print("Batch size: %d | Average latency per batch: %.3f ms | Throughput: %d" % (
            int(args.min_batch_size) * pow(2, i), latency_list[i], throughput_list[i]))

    # Write
    file_name = args.gpu_device.replace("/", "_")
    file_name = file_name + ".txt"
    with open("./results/" + file_name, "w") as f:
        for i in range(len(latency_list)):
            f.write("Batch size: %d | Average latency per batch: %.3f ms | Throughput: %d \n" % (
                int(args.min_batch_size) * pow(2, i), latency_list[i], throughput_list[i]))
