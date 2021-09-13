# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : kitti_util.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/5 上午11:06
@Desc   : Helper class and methods for processing KITTI object detection data
"""
import os
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay, ConvexHull


class KittiData:
    def __init__(self, dataset_path, file_idx, split='training'):
        self.calib_dir = os.path.join(dataset_path, split, "calib")
        self.image_dir = os.path.join(dataset_path, split, "image_2")
        self.lidar_dir = os.path.join(dataset_path, split, "velodyne")
        if split == "training":
            self.label_dir = os.path.join(dataset_path, split, "label_2")
        else:
            self.label_dir = None
        self.calib = self.get_calibration(file_idx)
        self.cv2_img = self.get_image(file_idx)
        self.img_width, self.img_height = self.get_image_shape(file_idx)
        self.lidar = self.get_lidar_data(file_idx)
        self.labels = self.get_label_info(file_idx)
        self.cls_gts = self.get_cls_groundtruths()
        self.bbox_color_dict = {"Vehicle": [1, 0, 0], "Pedestrian": [0, 1, 0], "Cyclist": [0, 0, 1], "GO": [0.5, 0.5, 0.5]}
        # self.pred_color_dict = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0.5, 0.5, 0.5]}
        self.cats = ["Vehicle", "Pedestrian", "Cyclist", "GO"]
        self.lines = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]

    def get_image(self, file_idx):
        image_name = str('%.6d.png' % file_idx)
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        return image

    def get_calibration(self, file_idx):
        calib_name = str('%.6d.txt' % file_idx)
        calib_path = os.path.join(self.calib_dir, calib_name)
        calib_dict = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                name, data = line.split(': ')
                data = np.array(data.strip('\n').split(' '), dtype=np.float32)
                if data.shape[0] == 9:
                    data = data.reshape([3, 3])
                elif data.shape[0] == 12:
                    data = data.reshape([3, 4])
                calib_dict[name] = data

        return calib_dict

    def get_image_shape(self, file_idx):
        image_name = str('%.6d.png' % file_idx)
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        width, height = image.size

        return width, height

    def get_lidar_data(self, file_idx):
        lidar_name = str('%.6d.bin' % file_idx)
        lidar_path = os.path.join(self.lidar_dir, lidar_name)
        lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        return lidar_data

    def get_label_info(self, file_idx):
        label_name = str('%.6d.txt' % file_idx)
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, "r") as f:
            label_info = [line.rstrip().split(" ") for line in f.readlines()]
            for label_list in label_info:
                label_list[1:] = [np.array(x, dtype=np.float32) for x in label_list[1:]]

        return label_info

    def get_R0(self):
        """
        Get R0_rect matrix

        :return: Rotation matrix R0_rect
        """
        assert 'R0_rect' in self.calib, 'No calibration file has been read !'
        return self.calib['R0_rect']

    def get_V2C(self):
        """
        Get Tr_velo_to_cam matrix

        :return: Projection matrix from velo to cam Tr_velo_to_cam
        """
        assert 'Tr_velo_to_cam' in self.calib, 'No calibration file has been read !'
        return self.calib['Tr_velo_to_cam']

    def get_P(self):
        """
        Get P2 matrix

        :return: Projection matrix P2
        """
        assert 'P2' in self.calib, 'No calibration file has been read !'
        return self.calib['P2']

    @staticmethod
    def cart2hom(pts_3d):
        """
        Transform coordinates from Cartesian coordinate system to homogeneous coordinate system

        :param pts_3d:Coordinates in Cartesian
        :return:Coordinates in homogeneous
        """
        N = pts_3d.shape[0]
        pts_3d_hom = np.concatenate([pts_3d, np.ones((N, 1))], axis=1)
        return pts_3d_hom

    @staticmethod
    def inverse_rigid_trans(Tr):
        """
        Inverse of rigid body transformation matrix

        :param Tr: rigid body transformation matrix
        :return:iTr : inverse Tr
        """
        iTr = np.zeros_like(Tr)  # 3x4
        iTr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        iTr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return iTr

    def project_rect_to_velo(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        velo coordinate system

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in velo coordinate system
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_rect_to_ref(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        camera 0 coordinate system

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in camera 0 coordinate system
        """
        return np.transpose(np.dot(np.linalg.inv(self.get_R0()), pts_3d_rect.T))

    def project_ref_to_velo(self, pts_3d_ref):
        """
        Coordinates transformation between ref and velo

        :param pts_3d_ref: Coordinates in ref coordinate system
        :return: Coordinates in velo coordinate system
        """
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.matmul(self.inverse_rigid_trans(self.get_V2C()), pts_3d_ref.T).T

    def project_velo_to_image(self, pts_3d_velo):
        """
        Coordinates transformation between velo coordinate system and
        image2

        :param pts_3d_rect: Coordinates in velo coordinate system
        :return: Coordinates in image 2
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_rect_to_image(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        image2

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in image 2
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        # pts_3d_rect2 = np.matmul(self.expand_R0(self.get_R0()),pts_3d_rect.T).T
        # pts_2d = np.matmul(self.get_P(), pts_3d_rect2.T).T
        pts_2d = np.matmul(self.get_P(), pts_3d_rect.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_rect(self, pts_3d_velo):
        """
        Coordinates transformation between velo coordinate system and
        camera 2 coordinate system

        :param pts_3d_velo: Coordinates in velo coordinate system
        :return: Coordinates in camera 2 coordinate system
        """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def project_velo_to_ref(self, pts_3d_velo):
        """
        Coordinates transformation between velo and ref

        :param pts_3d_velo: Coordinates in velo coordinate system
        :return: Coordinates in ref coordinate system
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.matmul(self.get_V2C(), pts_3d_velo.T).T

    def project_ref_to_rect(self, pts_3d_ref):
        """
        Coordinates transformation between camera 0 coordinate system and
        camera 2 coordinate system

        :param pts_3d_ref: Coordinates in camera 0 coordinate system
        :return: Coordinates in camera 2 coordinate system
        """
        return np.transpose(np.dot(self.get_R0(), np.transpose(pts_3d_ref)))

    def get_bbox(self, label):
        xyz = np.array(label[11:14])
        hwl = np.array(label[8:11])
        ry = label[14]
        r = R.from_euler("xyz", [0, ry, 0])
        r_matrix = r.as_matrix()

        bbox = np.zeros([8, 3])

        for i in range(8):
            if i & 1:
                bbox[i, 2] = xyz[2] + hwl[1] / 2.0
            else:
                bbox[i, 2] = xyz[2] - hwl[1] / 2.0
            if i & 2:
                bbox[i, 0] = xyz[0] + hwl[2] / 2.0
            else:
                bbox[i, 0] = xyz[0] - hwl[2] / 2.0
            if i & 4:
                bbox[i, 1] = xyz[1] + 0
            else:
                bbox[i, 1] = xyz[1] - hwl[0] / 1.0

        bbox = (r_matrix @ (bbox - xyz).T).T + xyz
        bbox = self.project_rect_to_velo(bbox)

        return bbox

    def get_cls_groundtruths(self):
        cls_gts = []
        for label in self.labels:
            if label[0] in ["Person_sitting", "Misc", "DontCare"]:
                continue

            cls_gt_dict = {}
            if label[0] in ["Car", "Van", "Truck", "Tram"]:
                cls_gt_dict["label"] = "Vehicle"
                cls_gt_dict["bbox"] = self.get_bbox(label)
            elif label[0] == "Pedestrian":
                cls_gt_dict["label"] = "Pedestrian"
                cls_gt_dict["bbox"] = self.get_bbox(label)
            elif label[0] == "Cyclist":
                cls_gt_dict["label"] = "Cyclist"
                cls_gt_dict["bbox"] = self.get_bbox(label)

            cls_gts.append(cls_gt_dict)

        return cls_gts

    def get_lidar_in_image_fov(self, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pcd_xyz_np = self.lidar[:, :3]
        pts_2d = self.project_velo_to_image(pcd_xyz_np)
        fov_inds = (
            (pts_2d[:, 0] < self.img_width)
            & (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 1] < self.img_height)
            & (pts_2d[:, 1] >= 0)
        )
        fov_inds = fov_inds & (pcd_xyz_np[:, 0] > clip_distance)
        pcd_fov_np = pcd_xyz_np[fov_inds, :]

        return pcd_fov_np

    def extract_lidar_in_image_fov(self, pcd, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pcd_xyz_np = np.asarray(pcd.points)
        pts_2d = self.project_velo_to_image(pcd_xyz_np)
        fov_inds = (
            (pts_2d[:, 0] < self.img_width)
            & (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 1] < self.img_height)
            & (pts_2d[:, 1] >= 0)
        )
        fov_inds = fov_inds & (pcd_xyz_np[:, 0] > clip_distance)
        pcd_fov_np = pcd_xyz_np[fov_inds, :]

        return pcd_fov_np

    def visualize_gts(self):
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.lidar[:, :3]))
        pcd.paint_uniform_color([0, 0, 0])

        line_sets = []
        for cls_gt in self.cls_gts:
            points = cls_gt["bbox"]
            color = self.bbox_color_dict[cls_gt["label"]]
            inlier_mask = self.in_hull(self.lidar[:, :3], points)
            inlier_indices = np.where(inlier_mask == True)
            print(f"{cls_gt['label']}: {len(inlier_indices[0])} points")
            pcd_colors = np.asarray(pcd.colors)
            pcd_colors[inlier_indices] = color
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                            lines=o3d.utility.Vector2iVector(self.lines))
            line_set.paint_uniform_color(color)
            line_sets.append(line_set)

        o3d.visualization.draw_geometries([pcd, *line_sets], window_name="Classification groundtruths in point cloud")

    def visualize_gts_img_fov(self):
        lidar_xyz_img_fov = self.get_lidar_in_image_fov()
        pcd_img_fov = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(lidar_xyz_img_fov))
        pcd_img_fov.paint_uniform_color([0, 0, 0])
        image = self.cv2_img.copy()
        line_sets = []
        for cls_gt in self.cls_gts:
            points = cls_gt["bbox"]
            points_2d = self.project_velo_to_image(points)
            color = self.bbox_color_dict[cls_gt["label"]]
            color_rgb = [c * 255 for c in color]
            color_bgr = [color_rgb[2], color_rgb[1], color_rgb[0]]
            for line in self.lines:
                image = cv2.line(image, points_2d[line[0]].astype(int).tolist(), points_2d[line[1]].astype(int).tolist(), color_bgr, thickness=2)
            inlier_mask = self.in_hull(lidar_xyz_img_fov, points)
            inlier_indices = np.where(inlier_mask == True)
            pcd_img_fov_colors = np.asarray(pcd_img_fov.colors)
            pcd_img_fov_colors[inlier_indices] = color
            pcd_img_fov.colors = o3d.utility.Vector3dVector(pcd_img_fov_colors)
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                            lines=o3d.utility.Vector2iVector(self.lines))
            line_set.paint_uniform_color(color)
            line_sets.append(line_set)

        o3d.visualization.draw_geometries([pcd_img_fov, *line_sets], window_name="Classification groundtruths in image fov")
        cv2.imshow("Classification groundtruths in image fov", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0

    @staticmethod
    def minimum_bounding_rectangle(points_xy):
        """
        Find the smallest bounding rectangle for a set of points.
        Returns a set of points representing the corners of the bounding box.

        :param points: an nx2 matrix of coordinates
        :return: an nx2 matrix of coordinates
        """
        pi2 = np.pi / 2

        # get the convex hull for the points
        hull = ConvexHull(points_xy)
        hull_points = points_xy[hull.vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points) - 1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            -np.sin(angles),
            np.sin(angles),
            np.cos(angles)
        ]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x_max = max_x[best_idx]
        x_min = min_x[best_idx]
        y_max = max_y[best_idx]
        y_min = min_y[best_idx]
        r_mat = rotations[best_idx]
        best_angle = angles[best_idx]

        best_box = np.zeros((4, 2))
        best_box[0] = np.dot([x_min, y_max], r_mat)
        best_box[1] = np.dot([x_max, y_max], r_mat)
        best_box[2] = np.dot([x_min, y_min], r_mat)
        best_box[3] = np.dot([x_max, y_min], r_mat)
        xy_center = np.dot([x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2], r_mat)
        l = y_max - y_min
        w = x_max - x_min
        rz_mat = np.array([[np.cos(best_angle), np.sin(best_angle), 0], [-np.sin(best_angle), np.cos(best_angle), 0], [0, 0, 1]], dtype=np.float32)
        rot = R.from_matrix(rz_mat)
        rz_euler = rot.as_euler("xyz")
        rz = rz_euler[2]

        return best_box, hull, w, l, rz, xy_center

    def fit_bbox_3d(self, pcd_cluster_np):
        z_min = pcd_cluster_np.min(axis=0)[-1]
        z_max = pcd_cluster_np.max(axis=0)[-1]
        h = z_max - z_min
        box_2d, hull, w, l, rz, xy_center = self.minimum_bounding_rectangle(pcd_cluster_np[:, :2])
        box_top_3d = np.hstack([box_2d, np.ones([box_2d.shape[0], 1]) * z_max])
        box_bottom_3d = np.hstack([box_2d, np.ones([box_2d.shape[0], 1]) * z_min])
        box_3d = np.vstack([box_top_3d, box_bottom_3d])
        xyz_center = np.append(xy_center, z_min)
        x_c = xyz_center[0]
        y_c = xyz_center[1]
        z_c = xyz_center[2]

        return box_3d, h, w, l, x_c, y_c, z_c, rz
