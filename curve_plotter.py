#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert_16")  # Model name
args = parser.parse_args()


def read_txt(txt_path):
    # MIG 4g.20gb | MIG 2g.10gb | MIG 1g.5gb
    qps_list = [[], [], []]
    latency_list = [[], [], []]

    with open(txt_path, "r") as f:
        lines = f.readlines()
        device_marker = 0
        for i in range(len(lines)):
            line = lines[i]
            line = line.replace(" ", "")
            line = line.replace("-", "")
            line = line.replace("\n", "")
            str_list = line.split("|")
            if len(str_list) > 1:
                throughput = float(str_list[1].split(":")[1].split("(")[0])
                latency = float(str_list[2].split(":")[1].split("(")[0])
                qps_list[device_marker].append(throughput)
                latency_list[device_marker].append(latency)
                if i < len(lines) - 1 and lines[i+1][0] != "B":
                    device_marker += 1
    f.close()
    return qps_list, latency_list


def format_func(value, tick_number):
    return pow(2, value)


if __name__ == '__main__':
    # Check file path
    file_name = "performance_summary.txt"
    if not os.path.exists(file_name):
        print("")
        print("Error:")
        print("--------------------------------------------------------------------")
        print("File name does not exist: %s ..." % file_name)
        print("--------------------------------------------------------------------")
        print("")
        exit(1)
    m_qps_list, m_latency_list = read_txt("performance_summary.txt")
    # Check output path
    if not os.path.exists("./output_figs"):
        os.mkdir("./output_figs")

    fig, ax = plt.subplots()
    x = [i for i in range(8)]
    ax.plot(x, m_qps_list[0], label='MIG 1g.5gb')
    ax.plot(x, m_qps_list[1], label='MIG 2g.10gb')
    ax.plot(x, m_qps_list[2], label='MIG 4g.20gb')
    print(plt.xlim([0, 7]))

    ax.grid(True)
    ax.legend(frameon=False)
    plt.xlabel("Batch_size")
    plt.ylabel("QPS")
    plt.title("QPS for " + args.model_name + " in Different Batch Size")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    file_path = "./output_figs/qps_" + args.model_name + ".png"
    plt.savefig(file_path)
    plt.show()

    fig, ax = plt.subplots()
    x = [i for i in range(8)]
    ax.plot(x, m_latency_list[0], label='MIG 1g.5gb')
    ax.plot(x, m_latency_list[1], label='MIG 2g.10gb')
    ax.plot(x, m_latency_list[2], label='MIG 4g.20gb')
    print(plt.xlim([0, 7]))

    ax.grid(True)
    ax.legend(frameon=False)
    plt.xlabel("Batch_size")
    plt.ylabel("Latency(ms)")
    plt.title("Latency for " + args.model_name + " in Different Batch Size")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    file_path = "./output_figs/latency_" + args.model_name + ".png"
    plt.savefig(file_path)
    plt.show()
