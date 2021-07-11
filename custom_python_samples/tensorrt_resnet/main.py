#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import onnx
import torch
import torchvision
import netron
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--layer_num", default="50")  # GPU Device
args = parser.parse_args()

resnet = ""
if args.layer_num == "50":
    resnet = torchvision.models.resnet50(pretrained=True).cuda()
elif args.layer_num == "101":
    resnet = torchvision.models.resnet101(pretrained=True).cuda()
elif args.layer_num == "152":
    resnet = torchvision.models.resnet152(pretrained=True).cuda()
else:
    print("Wrong layer num in ResNet (optional: 50, 101, 152) ... exit")
    exit(1)

# Check output path
if not os.path.exists("./onnx"):
    os.mkdir("./onnx")

onnx_path = "./onnx/resnet_" + args.layer_num + ".onnx"
x = torch.onnx.export(resnet,
                      torch.randn(1, 3, 224, 224, device='cuda'),
                      onnx_path,
                      verbose=False,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=13,
                      do_constant_folding=True,
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}, }
                      )
