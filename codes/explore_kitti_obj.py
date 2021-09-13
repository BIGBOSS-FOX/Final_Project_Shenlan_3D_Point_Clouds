# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : explore_kitti_obj.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/2 下午2:02
@Desc   : Visualize KITTI classification data ("Vehicle": red, "Pedestrian": green, "Cyclist": blue) from KITTI object detection data
"""
import argparse
import open3d as o3d

from kitti_util import KittiData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../../data/kitti")
    parser.add_argument("--file_idx", type=int, default=5876)

    return parser.parse_args()


def main():
    args = parse_args()
    kd = KittiData(dataset_path=args.root_dir, file_idx=args.file_idx)
    print(f"Visualizing classification groundtruths in original point cloud in {str('%.6d.bin' % args.file_idx)}...")
    kd.visualize_gts()
    print(f"Visualizing classification groundtruths in image fov in {str('%.6d.bin' % args.file_idx)}...")
    kd.visualize_gts_img_fov()
    print("Done!")


if __name__ == '__main__':
    main()
