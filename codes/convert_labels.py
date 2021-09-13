# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : convert_labels.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/9 下午2:55
@Desc   : Generate new labels by merging "Car", "Van", "Truck", "Tram" into "Vehicle"
"""
import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_label_dir",
                        type=str,
                        default="../../data/kitti/training/label_2",
                        help="directory that holds original labels")
    parser.add_argument("--dst_label_dir",
                        type=str,
                        default="../../data/kitti/training/my_label_2",
                        help="directory that stores converted labels")

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Converting labels from {args.src_label_dir} to {args.dst_label_dir} ...")
    os.makedirs(args.dst_label_dir, exist_ok=True)
    label_txt_names = os.listdir(args.src_label_dir)
    for label_txt_name in tqdm(label_txt_names):
        label_txt_path = os.path.join(args.src_label_dir, label_txt_name)
        with open(label_txt_path, "r") as f:
            label_info = [line.rstrip().split(" ") for line in f.readlines()]

        label_lines = []
        for l in label_info:
            if l[0] in ["Car", "Van", "Truck", "Tram"]:
                l[0] = "Vehicle"
            line = " ".join(l) + "\n"
            label_lines.append(line)

        save_txt_path = os.path.join(args.dst_label_dir, label_txt_name)
        with open(save_txt_path, "a") as f:
            for line in label_lines:
                f.write(line)


if __name__ == '__main__':
    main()
