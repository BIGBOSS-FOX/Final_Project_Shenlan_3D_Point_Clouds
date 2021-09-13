# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : ground_segmentation_object_clustering.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/7/19 16:43
@Desc   : Step 1. Remove the ground from the lidar points. Visualize ground as blue. (RANSAC)
          Step 2. Clustering over the remaining points. Visualize the clusters with random colors. (DBSCAN)
"""
import numpy as np
import open3d as o3d
import os
import struct
import argparse
from collections import deque
from tqdm import tqdm
from itertools import cycle, islice

from octree import octree_construction, octree_radius_search_fast
from result_set import RadiusNNResultSet


class RANSAC:
    """RANSAC algorithm

    Fit a plane model and use it to find the ground in the point cloud

    Attributes:
        inlier_dist (float): distance threshold for inliers
        n_points (int): a subsample of points to determine a plane, default: 3
        p (float): probability that at least one random sample is free from outliers
        plane_normal (np.ndarray): normal vector of the plane model
        plane_point (np.ndarray): a point on the plane model
        inlier_count (int): number of points that within the distance threshold from the model plane
        max_trails (int): maximum number of trails determined by p, outlier_ratio and n_points
    """
    def __init__(self, inlier_dist, p, outlier_ratio):
        self.inlier_dist = inlier_dist
        self.n_points = 3
        self.p = p
        self.plane_normal = np.empty((3,))
        self.plane_point = np.empty((3,))
        self.inlier_count = 0
        self.max_trials = int(np.log(1 - p) / np.log(1 - (1 - outlier_ratio) ** self.n_points))

    def fit(self, X):
        """Iteration to find the plane parameters (plane_normal and plane_point) so that the plane has the most
        inliers.

        Args:
            X (np.ndarray): point data
        """
        for _ in range(self.max_trials):
            # For each iteration, randomly sample 3 points
            R = X[np.random.choice(X.shape[0], self.n_points), :]
            # Plane normal vector is the cross product of two vectors constructed by the 3 points
            plane_normal = np.cross(R[2] - R[0], R[1] - R[0])
            # Skip if 3 points are collinear
            if np.linalg.norm(plane_normal) == 0:
                continue
            # Plane point is the joint point of two vectors
            plane_point = R[0]
            # Calculate distances from all points to the plane
            dists = np.abs(plane_normal @ (X - plane_point).T / np.linalg.norm(plane_normal))
            # Record inliers
            inlier_mask = np.where(dists < self.inlier_dist, True, False)
            inlier_count = np.count_nonzero(inlier_mask)
            # Update plane model parameters if the inlier count has exceed the maximum inlier count record
            if inlier_count > self.inlier_count:
                self.inlier_count = inlier_count
                self.plane_normal = plane_normal
                self.plane_point = plane_point

    def predict(self, X):
        """Predict inlier points given point cloud data

        Args:
            X (np.ndarray): point data

        Returns:
            inlier_mask (List[bool]): a boolean list indicating whether each point is an inlier or not
        """
        dists = np.abs(self.plane_normal @ (X - self.plane_point).T / np.linalg.norm(self.plane_normal))
        inlier_mask = np.where(dists < self.inlier_dist, True, False)
        return inlier_mask

    def LSQ(self, X, inlier_mask):
        """Run LSQ to refine the model after selecting the final model and inlier points

        Args:
            X (np.ndarray): point data
            inlier_mask (List[bool]): a boolean list indicating whether each point is an inlier or not

        Returns:
            new_inlier_mask (List[bool]): Updated inlier_mask
        """
        ransac_inliers = X[inlier_mask]
        A = np.hstack((ransac_inliers[:, :-1], np.ones((ransac_inliers.shape[0], 1))))
        b = ransac_inliers[:, -1]
        x_hat = np.linalg.inv((A.T @ A)) @ A.T @ b
        plane_params = np.array([x_hat[0], x_hat[1], -1, x_hat[2]])
        dists = np.abs(np.hstack((X, np.ones((X.shape[0], 1)))) @ plane_params) / np.linalg.norm(plane_params[:-1])
        new_inlier_mask = np.where(dists < self.inlier_dist, True, False)
        return new_inlier_mask


class DBSCAN:
    """DBSCAN algorithm

    RadiusNN search on an octree to determine whether a point is 'noise', 'border' or 'core'

    Attributes:
        r (float): radius
        min_samples (int): minimum number of neighbors required to be a core point
        leaf_size (int): leaf size of an octant
        min_extent (float): minimum extent of an octant
    """
    def __init__(self, r, min_samples, leaf_size=4, min_extent=0.01):
        self.r = r
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.min_extent = min_extent

    def fit(self, X):
        """Iterate until find a best fitted plane model for the ground

        Args:
            X (np.ndarray): point data
        """
        # Construct an octree
        root = octree_construction(X, leaf_size=self.leaf_size, min_extent=self.min_extent)
        # Initialize all points visited as False, labels as None and cluster_indices as -1
        self.visited = [False] * X.shape[0]
        self.labels = [None] * X.shape[0]
        self.cluster_indices = [-1] * X.shape[0]
        cur_cluster_index = 0
        # Initialize an empty queue to store neighbors
        neighbor_queue = deque()
        # For each point in point cloud
        for i, x in tqdm(enumerate(X), total=X.shape[0]):
            # Bypass if it is visited
            if self.visited[i]:
                continue
            # Otherwise mark it as visited
            self.visited[i] = True
            # RadiusNN search on the selected point
            result_set = RadiusNNResultSet(radius=self.r)
            octree_radius_search_fast(root, X, result_set, x)
            # if number of neighbors is less than min_samples threshold, label it as 'noise'
            if len(result_set.dist_index_list) < self.min_samples:
                self.labels[i] = 'noise'
                continue
            # Otherwise label it as 'core'
            self.labels[i] = 'core'
            # Assign point to current cluster index
            self.cluster_indices[i] = cur_cluster_index
            # Push all neighbors to the queue
            neighbor_queue.extend(result_set.dist_index_list[1:])
            # Iterate until the queue is empty
            while len(neighbor_queue) != 0:
                # Pop the left first neighbor from the queue
                neighbor_index = neighbor_queue.popleft().index
                # If the point has been previously labeled as 'noise', change it to 'border' and assign the cluster index
                if self.labels[neighbor_index] == 'noise':
                    self.labels[neighbor_index] = 'border'
                    self.cluster_indices[neighbor_index] = cur_cluster_index
                    continue
                # Bypass the point if it has been labeled as 'core' or 'border'
                if self.labels[neighbor_index] == 'core' or self.labels[neighbor_index] == 'border':
                    continue
                # For the unvisited point, mark it as visited
                self.visited[neighbor_index] = True
                # RadiusNN search on the selected point
                result_set = RadiusNNResultSet(radius=self.r)
                octree_radius_search_fast(root, X, result_set, X[neighbor_index])
                # If the number of neighbors is less than the min_samples, label it as 'border' and assign the cluster index
                if len(result_set.dist_index_list) < self.min_samples:
                    self.labels[neighbor_index] = 'border'
                    self.cluster_indices[neighbor_index] = cur_cluster_index
                # Else, label it as 'core', assign the cluster index and add its neighbors to the queue
                else:
                    self.labels[neighbor_index] = 'core'
                    self.cluster_indices[neighbor_index] = cur_cluster_index
                    neighbor_queue.extend(result_set.dist_index_list[1:])

            # Increment the cluster index when the neighbor queue is empty
            cur_cluster_index += 1

    def predict(self):
        """Return the cluster indices"""
        return self.cluster_indices


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_path",
                        type=str,
                        default="../data/0000000000.bin",
                        help="point cloud bin file")
    parser.add_argument("--inlier_dist",
                        type=float,
                        default=0.4,
                        help="distance threshold in RANSAC")
    parser.add_argument("--p",
                        type=float,
                        default=0.99,
                        help="probability that at least one random sample is free from outliers in RANSAC")
    parser.add_argument("--outlier_ratio",
                        type=float,
                        default=0.5,
                        help="proportion of outliers in RANSAC")
    parser.add_argument("--r",
                        type=float,
                        default=1,
                        help="radius in DBSCAN")
    parser.add_argument("--min_samples",
                        type=int,
                        default=4,
                        help="minimum samples in DBSCAN")
    args = parser.parse_args()
    bin_path, inlier_dist, p, outlier_ratio, r, min_samples = args.bin_path, \
                                                              args.inlier_dist, \
                                                              args.p, \
                                                              args.outlier_ratio, \
                                                              args.r, \
                                                              args.min_samples

    # Read point cloud data as a numpy ndarray
    pcd_np = read_velodyne_bin(bin_path)

    # Use RANSAC to find the points which belong to the ground
    print("Segmenting ground w/ RANSAC...")
    ransac = RANSAC(inlier_dist, p, outlier_ratio)
    ransac.fit(pcd_np)
    inlier_mask = ransac.predict(pcd_np)

    # Use LSQ to refine the ground plane model and update the inliers
    new_inlier_mask = ransac.LSQ(pcd_np, inlier_mask)

    # Visualize ground as blue
    print("Visualizing ground points as blue...")
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_np))
    pcd_colors = np.tile([0.5, 0.5, 0.5], (pcd_np.shape[0], 1))
    pcd_colors[new_inlier_mask] = np.array([0, 0, 1])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    o3d.visualization.draw_geometries([pcd])

    # Remove ground points and visualize the remaining points
    print("Removing the ground from the point clouds...")
    outlier_mask = np.logical_not(new_inlier_mask)
    pcd_remain_np = pcd_np[outlier_mask]
    print("Visualizing the remaining points...")
    pc_remain_colors = np.tile([0.5, 0.5, 0.5], (pcd_remain_np.shape[0], 1))
    pcd_remain = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd_remain_np))
    pcd_remain.colors = o3d.utility.Vector3dVector(pc_remain_colors)
    o3d.visualization.draw_geometries([pcd_remain])

    # Cluster over the remaining points
    print("Clustering over the remaining points w/ DBSCAN (this may take a long time)...")
    dbscan = DBSCAN(r, min_samples)
    dbscan.fit(pcd_remain_np)
    cluster_indices = dbscan.predict()

    # Visualize clusters with random colors (black points are considered as noise)
    print("Visualizing clusters...")
    colors = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    colors_mapping = np.array(list(islice(cycle(colors), max(cluster_indices))))
    colors_mapping = np.append(colors_mapping, [[0, 0, 0]], axis=0)
    cluster_colors = colors_mapping[cluster_indices]
    pcd_remain.colors = o3d.utility.Vector3dVector(cluster_colors)
    o3d.visualization.draw_geometries([pcd_remain])
    print("Done!")


if __name__ == '__main__':
    main()