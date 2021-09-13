# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : extract_kitti_cls.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/5 下午3:01
@Desc   : Extract KITTI classification data ("Vehicle", "Pedestrian", "Cyclist", "GO") from KITTI object detection data
"""
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

from kitti_util import KittiData
from ground_segmentation_object_clustering import RANSAC, DBSCAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_obj_dir", type=str, default="../../data/kitti")
    parser.add_argument("--kitti_cls_dir", type=str, default="../../data/kitti_cls")
    parser.add_argument("--veh_inlier_thres", type=int, default=200)
    parser.add_argument("--ped_inlier_thres", type=int, default=50)
    parser.add_argument("--cyc_inlier_thres", type=int, default=10)
    parser.add_argument("--go_count_lmt", type=int, default=5000)
    parser.add_argument("--ransac_dist_thres",
                        type=float,
                        default=0.2,
                        help="distance_threshold param in segment_plane")
    parser.add_argument("--ransac_n",
                        type=int,
                        default=3,
                        help="ransac_n param in segment_plane")
    parser.add_argument("--ransac_num_iters",
                        type=int,
                        default=1000,
                        help="num_iterations param in segment_plane")
    # parser.add_argument("--inlier_dist",
    #                     type=float,
    #                     default=0.1,
    #                     help="distance threshold in RANSAC")
    # parser.add_argument("--p",
    #                     type=float,
    #                     default=0.99,
    #                     help="probability that at least one random sample is free from outliers in RANSAC")
    # parser.add_argument("--outlier_ratio",
    #                     type=float,
    #                     default=0.5,
    #                     help="proportion of outliers in RANSAC")
    parser.add_argument("--dbscan_eps",
                        type=float,
                        default=1,
                        help="eps param in cluster_dbscan")
    parser.add_argument("--dbscan_min_points",
                        type=int,
                        default=50,
                        help="min_points param in cluster_dbscan")
    # parser.add_argument("--r",
    #                     type=float,
    #                     default=1,
    #                     help="radius in DBSCAN")
    # parser.add_argument("--min_samples",
    #                     type=int,
    #                     default=4,
    #                     help="minimum samples in DBSCAN")

    return parser.parse_args()


def remove_ground_plane_own(pcd, inlier_dist, p, outlier_ratio):
    pcd_np = np.asarray(pcd.points)
    # Use RANSAC to find the points which belong to the ground
    # print("Segmenting ground w/ RANSAC...")
    ransac = RANSAC(inlier_dist, p, outlier_ratio)
    ransac.fit(pcd_np)
    inlier_mask = ransac.predict(pcd_np)

    # Use LSQ to refine the ground plane model and update the inliers
    new_inlier_mask = ransac.LSQ(pcd_np, inlier_mask)

    outlier_mask = np.logical_not(new_inlier_mask)
    outlier_idxs = np.where(outlier_mask == True)[0]
    pcd_no_ground = pcd.select_by_index(outlier_idxs)

    return pcd_no_ground


def get_clusters_own(pcd, r, min_samples):
    pcd_np = np.asarray(pcd.points)
    # print("Clustering over the remaining points w/ DBSCAN (this may take a long time)...")
    dbscan = DBSCAN(r, min_samples)
    dbscan.fit(pcd_np)
    labels = dbscan.predict()
    max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels


def remove_ground_plane_api(pcd, dist_thres, ransac_n, num_iters):
    # print("=" * 64)
    # print("Removing ground plane from point cloud using RANSAC ...")
    pcd_np = np.array(pcd.points)

    pcd_normals_np = np.asarray(pcd.normals)
    # keep points with angle between their normals and z-axis less than 30 degree for RANSAC
    min_angular = np.cos(np.pi / 6)
    pcd_normal_z_angular_np = np.abs(pcd_normals_np[:, 2])
    pcd_roi_np = pcd_np[pcd_normal_z_angular_np > min_angular]
    pcd_rest_np = pcd_np[np.logical_not(pcd_normal_z_angular_np > min_angular)]
    pcd_roi = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_roi_np))

    plane_model, inliers = pcd_roi.segment_plane(distance_threshold=dist_thres,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iters)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    pcd_outliers = pcd_roi.select_by_index(inliers, invert=True)
    pcd_outliers_np = np.asarray(pcd_outliers.points)
    pcd_no_plane_np = np.vstack([pcd_rest_np, pcd_outliers_np])
    pcd_no_plane = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_no_plane_np))

    return pcd_no_plane


def get_clusters_api(pcd, eps, min_points):
    # print("=" * 64)
    # print("Clustering point cloud using DBSCAN ...")
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels


def extract_cls_data(kitti_obj_dir, kitti_cls_dir, split, args):
    kitti_cls_data_dir = os.path.join(kitti_cls_dir, split)
    os.makedirs(os.path.join(kitti_cls_data_dir, "Vehicle"), exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_data_dir, "Pedestrian"), exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_data_dir, "Cyclist"), exist_ok=True)
    os.makedirs(os.path.join(kitti_cls_data_dir, "GO"), exist_ok=True)
    split_txt = os.path.join(kitti_obj_dir, f"ImageSets/{split}.txt")
    split_idxs = np.loadtxt(split_txt, dtype=int)

    print(f"Extracting kitti_cls {split} data to {kitti_cls_data_dir}...")
    vehicle_count = 0
    pedestrian_count = 0
    cyclist_count = 0
    go_count = 0

    for idx in tqdm(split_idxs, total=len(split_idxs)):
        kd = KittiData(dataset_path=kitti_obj_dir, file_idx=idx)
        inlier_mask = np.zeros(kd.lidar[:, :3].shape[0], dtype=bool)
        for cls_gt in kd.cls_gts:
            label = cls_gt["label"]
            inlier = kd.in_hull(kd.lidar[:, :3], cls_gt["bbox"])
            inlier_mask = inlier_mask | inlier
            # If number of inliers is below certain threshold, discard it
            if label == "Vehicle" and np.count_nonzero(inlier) < args.veh_inlier_thres:
                continue
            if label == "Pedestrian" and np.count_nonzero(inlier) < args.ped_inlier_thres:
                continue
            if label == "Cyclist" and np.count_nonzero(inlier) < args.cyc_inlier_thres:
                continue

            lidar_xyz_cls = kd.lidar[:, :3][inlier]

            if label == "Vehicle":
                np.savetxt(os.path.join(kitti_cls_data_dir, label, f"{vehicle_count}.txt"), lidar_xyz_cls)
                vehicle_count += 1
            elif label == "Pedestrian":
                np.savetxt(os.path.join(kitti_cls_data_dir, label, f"{pedestrian_count}.txt"), lidar_xyz_cls)
                pedestrian_count += 1
            elif label == "Cyclist":
                np.savetxt(os.path.join(kitti_cls_data_dir, label, f"{cyclist_count}.txt"), lidar_xyz_cls)
                cyclist_count += 1

        if go_count >= args.go_count_lmt:
            continue

        outlier_mask = np.logical_not(inlier_mask)
        lidar_xyz_no_gts = kd.lidar[:, :3][outlier_mask]
        pcd_no_gts = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(lidar_xyz_no_gts))
        # o3d.visualization.draw_geometries([pcd_no_gts])

        pcd_img_fov_no_gts_np = kd.extract_lidar_in_image_fov(pcd_no_gts)
        pcd_img_fov_no_gts = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_img_fov_no_gts_np))
        pcd_img_fov_no_gts.estimate_normals()
        # o3d.visualization.draw_geometries([pcd_img_fov_no_gts])

        # pcd_img_fov_no_ground = remove_ground_plane_own(pcd_img_fov_no_gts, args.inlier_dist, args.p, args.outlier_ratio)
        pcd_img_fov_no_ground = remove_ground_plane_api(pcd_img_fov_no_gts, args.ransac_dist_thres, args.ransac_n, args.ransac_num_iters)
        # o3d.visualization.draw_geometries([pcd_img_fov_no_ground])

        # Own implementation of DBSCAN is too slow
        # pcd_img_fov_no_ground, cluster_idxs = get_clusters_own(pcd_img_fov_no_ground, args.r, args.min_samples)
        pcd_img_fov_no_ground, cluster_idxs = get_clusters_api(pcd_img_fov_no_ground, args.dbscan_eps, args.dbscan_min_points)
        # o3d.visualization.draw_geometries([pcd_img_fov_no_ground])

        pcd_img_fov_no_ground_np = np.asarray(pcd_img_fov_no_ground.points)

        for cluster_idx in range(cluster_idxs.max() + 1):
            lidar_xyz_go = pcd_img_fov_no_ground_np[np.where(cluster_idxs == cluster_idx)]
            np.savetxt(os.path.join(kitti_cls_data_dir, "GO", f"{go_count}.txt"), lidar_xyz_go)
            go_count += 1


def main():
    args = parse_args()
    extract_cls_data(args.kitti_obj_dir, args.kitti_cls_dir, "train", args)
    extract_cls_data(args.kitti_obj_dir, args.kitti_cls_dir, "val", args)


if __name__ == '__main__':
    main()

