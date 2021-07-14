#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert_16")  # Model name
args = parser.parse_args()


def get_info_dict(log_dir_path):
    # Info dict
    info = {}

    file_names = os.listdir(log_dir_path)
    for file_name in file_names:
        if file_name[-4:] == ".txt":
            # Throughput
            throughput = 0
            # Latency
            latency = 0
            # Get batch size
            str_list = file_name.split("_")[3]
            batch_size = int(str_list.split(".")[0])

            with open(log_dir_path + "/" + file_name, "r") as f:
                lines = f.readlines()
                for line in lines:
                    # Get Throughput
                    throughput_info = re.search("Throughput: ", line, re.M | re.I)
                    if throughput_info is not None:
                        tmp_throughput = float(line[throughput_info.span()[1]:].split(" ")[0])
                        throughput = tmp_throughput * float(batch_size)
                    # Get Latency (mean GPU Compute Time)
                    latency_info = re.search("GPU Compute Time: min", line, re.M | re.I)
                    if latency_info is not None:
                        str_list = line.split(" ")
                        if str_list[13] != "mean" or str_list[14] != "=":
                            print("Error might occurred in the match of str_list, please check the corresponding log info and modify this part of the code...")
                            exit(1)
                        latency = float(str_list[15])
                    # Get device info
                    device_info = re.search("Selected Device: ", line, re.M | re.I)
                    if device_info is not None:
                        device_name = line[device_info.span()[1]:].replace("\n", "")
                    # Wait for the ready of infos and update the info dict
                    if throughput > 0 and latency > 0:
                        # If read, update the info dict
                        if device_name in info.keys():
                            # Already has info about this device
                            info[device_name][batch_size] = [throughput, latency]
                        else:
                            # New in info dict
                            info[device_name] = {}
                            info[device_name][batch_size] = [throughput, latency]
    return info


def format_output(info):
    # Output list
    output_list = []

    # Sorted by dict.keys()
    keys_list = sorted(info.keys(), reverse=False)
    for key in keys_list:
        print("")
        output_list.append("")
        print("--------------------------------------------------------------------")
        output_list.append("--------------------------------------------------------------------")
        print("Device Name: %s" % key)
        output_list.append("Device Name: %s" % key)
        print("--------------------------------------------------------------------")
        output_list.append("--------------------------------------------------------------------")
        device_info = info[key]
        sub_key_list = sorted(device_info.keys(), reverse=False)
        for sub_key in sub_key_list:
            throughput = device_info[sub_key][0]
            latency = device_info[sub_key][1]
            print("Batch Size: %d | Throughput: %.2f (qps) | Latency: %.4f (ms)" % (sub_key, throughput, latency))
            output_list.append("Batch Size: %d | Throughput: %.2f (qps) | Latency: %.4f (ms)" % (sub_key, throughput, latency))
        print("--------------------------------------------------------------------")
        output_list.append("--------------------------------------------------------------------")

    # Write into txt
    with open("./performance_summary.txt", "w") as f:
        for message in output_list:
            f.write(message + "\n")
    f.close()


if __name__ == '__main__':
    info_dict = get_info_dict("./log/" + args.model_name)
    format_output(info_dict)
