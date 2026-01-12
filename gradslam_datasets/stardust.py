import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class StardustDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 660,
        desired_width: Optional[int] = 1280,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        self.associations, self.imus, self.times, self.image_data, self.depth_data = self.get_all_associated()
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(
        self,
        tstamp_image,
        tstamp_depth,
        tstamp_pose=None,
        tstamp_imu=None,
        max_dt=0.015,
    ):
        """pair images, depths, and poses using nearest neighbor"""
        associations = []
        lstart = 0
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None and tstamp_imu is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            elif tstamp_imu is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

            else:  # imu and pose provided
                lend = np.argmin(np.abs(tstamp_imu - t))
                l = np.arange(lstart, lend + 1, step=1)

                if (np.abs(tstamp_imu[lend] - t) < max_dt):
                    associations.append((i, i, i, l))
                    lstart = lend + 1

        return associations

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix() # x,y,z,w

        pose[:3, 3] = pvec[:3]
        return pose

    def get_filepaths(self):
        color_paths, depth_paths = [], []
        for i, j, k, l in self.associations:
            color_paths += [os.path.join(self.input_folder, "rgb", self.image_data[i, 1])]
            depth_paths += [os.path.join(self.input_folder, "depth", self.depth_data[j, 1])]

        embedding_paths = None

        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        for i, j, k, l in self.associations:
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]
        return poses

    def load_imu(self):
        # '''
        # Read IMU data. IMU data is in a .txt containing relative transformation for each time step
        # '''
        imu_meas = []
        for i, j, k, l in self.associations:
            imu_meas += [torch.from_numpy(self.imus[l, :]).float()]
        return imu_meas

    def load_tstamps(self):
        tstamps = []
        for i, j, k, l in self.associations:
            # tstamps += [torch.tensor([tstamp_image[i]]).float().to(self.device)]
            tstamps += [self.times[i]]
        return tstamps

    def get_c2i_tf(self):
        """Get the transformation matrix from camera optical frame to IMU frame"""
        tf_list = os.path.join(self.input_folder, "tf.txt")

        tf_data = self.parse_list(tf_list).astype(np.float64)

        # Convert translation+quaternion to homogeneous matrix
        i2c = self.pose_matrix_from_quaternion(tf_data)
        c2i = np.linalg.inv(i2c)

        return torch.from_numpy(c2i).float().to(self.device)

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

    def get_all_associated(self):
        image_list = os.path.join(self.input_folder, "rgb.txt")
        depth_list = os.path.join(self.input_folder, "depth.txt")
        imu_list = os.path.join(self.input_folder, "imu.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        imu_data = self.parse_list(imu_list)
        imu_vecs = imu_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = None
        tstamp_imu = imu_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose, tstamp_imu
        )

        return associations, imu_vecs, tstamp_image, image_data, depth_data
