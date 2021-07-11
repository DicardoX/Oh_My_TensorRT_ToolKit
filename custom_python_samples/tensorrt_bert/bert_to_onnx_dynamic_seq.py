#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import numpy as np
import torch
import onnxruntime
import argparse
import os

from models.bert_custom import BertModel_custom

parser = argparse.ArgumentParser()
parser.add_argument("--seq_len", default="16")  # Sequence length
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0"


def make_position_input(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
    return position_ids


def make_train_dummy_input(seq_len):
    org_input_ids = torch.LongTensor([[i for i in range(seq_len)]])
    org_token_type_ids = torch.LongTensor([[1 for i in range(seq_len)]])
    org_input_mask = torch.LongTensor([[0 for i in range(int(seq_len/2))] + [1 for i in range(seq_len - int(seq_len/2))]])
    org_position_ids = make_position_input(org_input_ids)
    return (org_input_ids, org_token_type_ids, org_input_mask, org_position_ids)


if __name__ == '__main__':
    MODEL_ONNX_PATH = "./onnx/bert_" + args.seq_len + ".onnx"
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX

    model = BertModel_custom.from_pretrained('bert-base-uncased')
    model.train(False)

    org_dummy_input = make_train_dummy_input(int(args.seq_len))

    output = torch.onnx.export(model,
                               org_dummy_input,
                               MODEL_ONNX_PATH,
                               verbose=True,
                               operator_export_type=OPERATOR_EXPORT_TYPE,
                               input_names=['input_ids', 'token_type_ids', 'attention_mask', 'position_ids'],
                               output_names=['output'],
                               do_constant_folding=True,
                               dynamic_axes={"input_ids": {0: "batch_size"}, "token_type_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "output": {0: "batch_size"},}
                               )
    print("Export of torch_model.onnx complete!")