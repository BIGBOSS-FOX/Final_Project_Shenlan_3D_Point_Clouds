# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : dataset.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/6 下午6:18
@Desc   : Define KITTI Classification dataset
"""
import torch
import os
import glob
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class KittiClsDataset(Dataset):
    def __init__(self, root_dir, augmentation=True, point_nums=1000):
        self.root_dir = root_dir
        self.cats = ["Vehicle", "Pedestrian", "Cyclist", "GO"]
        self.txt_paths = self._get_txt_paths()
        self.augmentation = augmentation
        self.point_nums = point_nums

    def _get_txt_paths(self):
        txt_paths = []
        for cat in self.cats:
            cat_dir = os.path.join(self.root_dir, cat)
            cat_txt_paths = glob.glob(f"{cat_dir}/*.txt")
            cat_txt_paths = [os.path.normpath(path) for path in cat_txt_paths]
            txt_paths += cat_txt_paths
        return txt_paths

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        pcd_np = np.loadtxt(self.txt_paths[idx], dtype=np.float32)
        cat = self.txt_paths[idx].split(os.sep)[-2]
        label_idx = self.cats.index(cat)
        if self.augmentation:

            if pcd_np.shape[0] > self.point_nums:
                random_point_idxs = np.random.randint(0, pcd_np.shape[0], size=self.point_nums)
                pcd_np = pcd_np[random_point_idxs]

            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pcd_np[:, [0, 1]] = pcd_np[:, [0, 1]] @ rotation_matrix
            pcd_np_std = pcd_np.std(axis=0)
            if np.any(pcd_np_std < 1e-6):
                pcd_np_std += 1e-6
            pcd_np = (pcd_np - pcd_np.mean(axis=0)) / pcd_np_std
            # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_np))
            # o3d.visualization.draw_geometries([pcd])

        label = torch.from_numpy(np.array(label_idx).astype(np.int64))
        pcd = torch.from_numpy(pcd_np.T.astype(np.float32))
        pcd = F.pad(pcd, pad=(0, self.point_nums - pcd.shape[1]), mode='constant', value=0)

        return pcd, label


if __name__ == '__main__':
    train_dir = "../../data/kitti_cls/train"
    train_set = KittiClsDataset(train_dir)

    vehicle_counts = len(os.listdir(os.path.join(train_dir, "Vehicle")))
    pedestrian_counts = len(os.listdir(os.path.join(train_dir, "Pedestrian")))
    cyclist_counts = len(os.listdir(os.path.join(train_dir, "Cyclist")))
    go_counts = len(os.listdir(os.path.join(train_dir, "GO")))

    # Apply WeightedRandomSampler to imbalanced datasets
    train_class_weights = [
        1 / vehicle_counts,
        1 / pedestrian_counts,
        1 / cyclist_counts,
        1 / go_counts
    ]

    train_sample_weights = np.zeros(len(train_set))
    train_sample_weights[:vehicle_counts] = train_class_weights[0]
    train_sample_weights[vehicle_counts:vehicle_counts + pedestrian_counts] = train_class_weights[1]
    train_sample_weights[vehicle_counts + pedestrian_counts:vehicle_counts + pedestrian_counts + cyclist_counts] = train_class_weights[2]
    train_sample_weights[vehicle_counts + pedestrian_counts + cyclist_counts:] = train_class_weights[3]

    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_set), replacement=True)

    train_loader_original = DataLoader(train_set, batch_size=10, shuffle=True)
    train_loader_weighted = DataLoader(train_set, batch_size=10, sampler=train_sampler)

    print("=" * 64)
    print(f"Run {len(train_set) // 10} batches from train_loader_original:")
    cnt = 0
    counter = {}
    for idx, (pts, label) in enumerate(train_loader_original):
        print(f"batch {idx} labels: {label}")
        label = label.tolist()
        for l in label:
            if l not in counter:
                counter[l] = 0
            counter[l] += 1
        cnt += 1
        if cnt > len(train_set) // 10:
            print(f"batch shape: {pts.shape}")
            print(f"label counter: {counter}")
            break

    print("=" * 64)
    print(f"Run {len(train_set) // 10} batches from train_loader_weighted:")
    cnt = 0
    counter = {}
    for idx, (pts, label) in enumerate(train_loader_weighted):
        print(f"batch {idx} labels: {label}")
        label = label.tolist()
        for l in label:
            if l not in counter:
                counter[l] = 0
            counter[l] += 1
        cnt += 1
        if cnt > len(train_set) // 10:
            print(f"batch shape: {pts.shape}")
            print(f"label counter: {counter}")
            break