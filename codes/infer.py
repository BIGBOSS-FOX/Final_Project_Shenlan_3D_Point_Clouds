# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : infer.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/9 上午10:42
@Desc   : Run inference on
            1. a single point cloud, if file_idx is provided, and visualize groundtruth and result
            2. all point clouds in val set and write results in KITTI format
"""
import argparse
import time
import os
import sys
import datetime
import numpy as np
import open3d as o3d
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from kitti_util import KittiData
from extract_kitti_cls import remove_ground_plane_api, get_clusters_api
from dataset import KittiClsDataset
from models import PointNet, VFE, VFE_LW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        type=str,
                        default="../../data/kitti",
                        help="dataset root dir")
    parser.add_argument("--file_idx",
                        type=int,
                        default=None,
                        help="Run inference on single point cloud and visualize result if file_idx is provided. Otherwise, write results in KITTI label format")
    parser.add_argument("--ckpt",
                        type=str,
                        default="../runs/train/PointNet/exp2/checkpoints/latest.tar",
                        help="checkpoint path")
    parser.add_argument("--result_dir",
                        type=str,
                        default="../kitti_eval/results",
                        help="directory that stores inference and evaluation results")
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
    parser.add_argument("--dbscan_eps",
                        type=float,
                        default=0.5,
                        help="eps param in cluster_dbscan")
    parser.add_argument("--dbscan_min_points",
                        type=int,
                        default=10,
                        help="min_points param in cluster_dbscan")
    parser.add_argument("--veh_h_min",
                        type=float,
                        default=1,
                        help="vehicle min height")
    parser.add_argument("--veh_h_max",
                        type=float,
                        default=2.5,
                        help="vehicle max height")
    parser.add_argument("--veh_len_min",
                        type=float,
                        default=1,
                        help="vehicle min length")
    parser.add_argument("--veh_len_max",
                        type=float,
                        default=10,
                        help="vehicle max length")
    parser.add_argument("--ped_h_min",
                        type=float,
                        default=1.2,
                        help="pedestrian min height")
    parser.add_argument("--ped_h_max",
                        type=float,
                        default=2,
                        help="pedestrian max height")
    parser.add_argument("--ped_len_min",
                        type=float,
                        default=0.2,
                        help="pedestrian min length")
    parser.add_argument("--ped_len_max",
                        type=float,
                        default=3,
                        help="pedestrian max length")
    parser.add_argument("--cyc_h_min",
                        type=float,
                        default=1.2,
                        help="cyclist min height")
    parser.add_argument("--cyc_h_max",
                        type=float,
                        default=2,
                        help="cyclist max height")
    parser.add_argument("--cyc_len_min",
                        type=float,
                        default=0.2,
                        help="cyclist min length")
    parser.add_argument("--cyc_len_max",
                        type=float,
                        default=3,
                        help="cyclist max length")
    return parser.parse_args()


def load_checkpoint(args):
    """Load model from checkpoint"""
    print(f"Loading model's checkpoint from {args.ckpt}")
    model_name = os.path.normpath(args.ckpt).split(os.sep)[-4]
    if model_name == "PointNet":
        model = PointNet().cuda()
    elif model_name == "VFE":
        model = VFE(point_nums=1000).cuda()
    else:
        model = VFE_LW(point_nums=1000).cuda()
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def TTA(pcd_np):
    """Test-Time Augmentations"""
    if pcd_np.shape[0] > 1000:
        random_point_idxs = np.random.randint(0, pcd_np.shape[0], size=1000)
        pcd_np = pcd_np[random_point_idxs]

    theta = np.random.uniform(0, np.pi * 2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pcd_np[:, [0, 1]] = pcd_np[:, [0, 1]] @ rotation_matrix
    pcd_np_std = pcd_np.std(axis=0)
    if np.any(pcd_np_std < 1e-6):
        pcd_np_std += 1e-6
    pcd_np = (pcd_np - pcd_np.mean(axis=0)) / pcd_np_std

    pcd_tensor = torch.from_numpy(pcd_np.T.astype(np.float32))
    pcd_tensor = F.pad(pcd_tensor, pad=(0, 1000 - pcd_tensor.shape[1]), mode='constant', value=0)

    return pcd_tensor


def infer(root_dir, file_idx, model, device, args, visualize=False):
    with torch.no_grad():
        kd = KittiData(root_dir, file_idx)
        if visualize:
            print(f"Visualizing classification groundtruths in image fov in {str('%.6d.bin' % file_idx)}...")
            kd.visualize_gts_img_fov()

        # Get pcd in img fov
        pcd_img_fov_np = kd.get_lidar_in_image_fov()
        pcd_img_fov = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_img_fov_np))
        pcd_img_fov.estimate_normals()
        # o3d.visualization.draw_geometries([pcd_img_fov])

        # Remove ground
        pcd_img_fov_no_ground = remove_ground_plane_api(pcd_img_fov, args.ransac_dist_thres, args.ransac_n,
                                                        args.ransac_num_iters)
        # o3d.visualization.draw_geometries([pcd_img_fov_no_ground])

        # Get clusters
        pcd_img_fov_no_ground, cluster_idxs = get_clusters_api(pcd_img_fov_no_ground, args.dbscan_eps,
                                                               args.dbscan_min_points)
        if visualize:
            print("Visualizing point cloud after removing ground and clustering...")
            o3d.visualization.draw_geometries([pcd_img_fov_no_ground], window_name="Clusters in point cloud")

        # Clusters to input_tensor_batch
        pcd_img_fov_no_ground_np = np.asarray(pcd_img_fov_no_ground.points)
        cluster_tensors = []
        for cluster_idx in range(cluster_idxs.max() + 1):
            pcd_cluster_np = pcd_img_fov_no_ground_np[np.where(cluster_idxs == cluster_idx)]
            pcd_cluster_tensor = TTA(pcd_cluster_np)
            cluster_tensors.append(pcd_cluster_tensor)

        input_tensor_batch = torch.stack(cluster_tensors, dim=0).to(device)

        # Run inference
        outs = model(input_tensor_batch)
        probs = F.softmax(outs, dim=1)
        confs, preds = torch.max(probs, 1)

        confs = confs.tolist()
        preds = preds.tolist()

        pred_results = []
        for i, (pred, conf) in enumerate(zip(preds, confs)):
            pcd_cluster_np = pcd_img_fov_no_ground_np[np.where(cluster_idxs == i)]
            try:
                pred_result = {}
                pred_result["pred"] = kd.cats[pred]
                pred_result["bbox_3d"], \
                pred_result["h_velo"], \
                pred_result["w_velo"], \
                pred_result["l_velo"], \
                pred_result["x_velo"], \
                pred_result["y_velo"], \
                pred_result["z_velo"], \
                pred_result["rz_velo"] = kd.fit_bbox_3d(pcd_cluster_np)
                pred_result["score"] = conf
                pred_results.append(pred_result)
            except:
                # print("Error in ConvexHull()")
                continue

        # print(pred_results)

        # Visualize raw inference result
        if visualize:
            pcd_img_fov_no_ground.paint_uniform_color([0, 0, 0])
            pcd_img_fov_no_ground_colors = np.asarray(pcd_img_fov_no_ground.colors)
            line_sets = []
            for i, pred_result in enumerate(pred_results):
                color = kd.bbox_color_dict[pred_result["pred"]]
                bbox_3d = pred_result["bbox_3d"]
                pcd_img_fov_no_ground_colors[np.where(cluster_idxs == i)] = color
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(bbox_3d),
                                                lines=o3d.utility.Vector2iVector(kd.lines))
                line_set.paint_uniform_color(color)
                line_sets.append(line_set)
            pcd_img_fov_no_ground.colors = o3d.utility.Vector3dVector(pcd_img_fov_no_ground_colors)
            print("Visualizing raw inference result...")
            o3d.visualization.draw_geometries([pcd_img_fov_no_ground, *line_sets], window_name="Raw inference result")

        # Post process on raw inference result
        for i, pred_result in enumerate(pred_results):
            if pred_result["pred"] == "Vehicle":
                if pred_result["h_velo"] < args.veh_h_min \
                        or pred_result["h_velo"] > args.veh_h_max \
                        or max(pred_result["l_velo"], pred_result["w_velo"]) > args.veh_len_max \
                        or min(pred_result["l_velo"], pred_result["w_velo"]) < args.veh_len_min:
                    pred_result["pred"] = "GO"
            elif pred_result["pred"] == "Pedestrian":
                if pred_result["h_velo"] < args.ped_h_min \
                        or pred_result["h_velo"] > args.ped_h_max \
                        or max(pred_result["l_velo"], pred_result["w_velo"]) > args.ped_len_max \
                        or min(pred_result["l_velo"], pred_result["w_velo"]) < args.ped_len_min \
                        or min(pred_result["l_velo"], pred_result["w_velo"]) > pred_result["h_velo"]:
                    pred_result["pred"] = "GO"
            elif pred_result["pred"] == "Cyclist":
                if pred_result["h_velo"] < args.cyc_h_min \
                        or pred_result["h_velo"] > args.cyc_h_max \
                        or max(pred_result["l_velo"], pred_result["w_velo"]) > args.cyc_len_max \
                        or min(pred_result["l_velo"], pred_result["w_velo"]) < args.cyc_len_min \
                        or min(pred_result["l_velo"], pred_result["w_velo"]) > pred_result["h_velo"]:
                    pred_result["pred"] = "GO"
        # print(pred_results)

        # Visualize processed inference result
        if visualize:
            pcd_img_fov_no_ground.paint_uniform_color([0, 0, 0])
            pcd_img_fov_no_ground_colors = np.asarray(pcd_img_fov_no_ground.colors)
            image = kd.cv2_img.copy()
            line_sets = []
            for i, pred_result in enumerate(pred_results):
                color = kd.bbox_color_dict[pred_result["pred"]]
                color_rgb = [c * 255 for c in color]
                color_bgr = [color_rgb[2], color_rgb[1], color_rgb[0]]
                bbox_3d = pred_result["bbox_3d"]
                bbox_2d = kd.project_velo_to_image(bbox_3d)
                for line in kd.lines:
                    image = cv2.line(image, bbox_2d[line[0]].astype(int).tolist(), bbox_2d[line[1]].astype(int).tolist(), color_bgr, thickness=2)
                pcd_img_fov_no_ground_colors[np.where(cluster_idxs == i)] = color
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(bbox_3d),
                                                lines=o3d.utility.Vector2iVector(kd.lines))
                line_set.paint_uniform_color(color)
                line_sets.append(line_set)
            pcd_img_fov_no_ground.colors = o3d.utility.Vector3dVector(pcd_img_fov_no_ground_colors)
            print("Visualizing processed inference result...")
            o3d.visualization.draw_geometries([pcd_img_fov_no_ground, *line_sets], window_name="Processed inference result")
            cv2.imshow("Processed inference result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return pred_results


def write_inference_result(pred_results, result_data_dir, file_idx, args):
    file_name = str('%.6d.txt' % file_idx)
    file_path = os.path.join(result_data_dir, file_name)
    kd = KittiData(args.root_dir, file_idx)
    with open(file_path, "a") as f:
        lines = []
        for pred_result in pred_results:
            if pred_result["pred"] == "GO":
                continue
            xyz_velo = np.array([[pred_result["x_velo"], pred_result["y_velo"], pred_result["z_velo"]]])
            xyz_rect = kd.project_velo_to_rect(xyz_velo)
            bbox_2d = kd.project_velo_to_image(pred_result["bbox_3d"])
            x_min, y_min = bbox_2d.min(axis=0)
            x_max, y_max = bbox_2d.max(axis=0)

            line_list = [
                pred_result["pred"],
                str(-1),
                str(-1),
                str(-10),
                '%.4f' % x_min,
                '%.4f' % y_min,
                '%.4f' % x_max,
                '%.4f' % y_max,
                '%.4f' % pred_result["h_velo"],
                '%.4f' % pred_result["w_velo"],
                '%.4f' % pred_result["l_velo"],
                '%.4f' % xyz_rect[0][0],
                '%.4f' % xyz_rect[0][1],
                '%.4f' % xyz_rect[0][2],
                '%.4f' % pred_result["rz_velo"],
                '%.4f' % pred_result["score"],
            ]
            line_str = " ".join(line_list) + "\n"
            lines.append(line_str)

        for line in lines:
            f.write(line)


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = load_checkpoint(args)
    model.eval()

    if args.file_idx is not None:
        pred_results = infer(args.root_dir, args.file_idx, model, device, args, visualize=True)
        print("Done!")
    else:
        val_split_txt = os.path.join(args.root_dir, "ImageSets", "val.txt")
        val_split_idxs = np.loadtxt(val_split_txt, dtype=int)

        result_data_dir = os.path.join(args.result_dir, "data")
        os.makedirs(result_data_dir, exist_ok=True)
        print(f"Writing inference results into {result_data_dir}")
        for val_split_idx in tqdm(val_split_idxs, total=len(val_split_idxs)):
            pred_results = infer(args.root_dir, val_split_idx, model, device, args, visualize=False)
            write_inference_result(pred_results, result_data_dir, val_split_idx, args)
        print("Done!")


if __name__ == '__main__':
    main()
