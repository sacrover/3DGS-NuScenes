import csv
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


class SensorParameters():

    def __init__(self, nusc: NuScenes, sample: dict, sensors: list):
        self.nusc = nusc
        self.current_sample = sample
        self.sensor_calib_data = None
        self.sensors = sensors
        self.T_gs = {}  # Tansform matrices
        self.T_sg = {}  # Tansform matrices

        # sensor extrinsic parameters with respect to ego vehicle body frame
        self.RADAR_FRONT = None
        self.RADAR_FRONT_LEFT = None
        self.RADAR_FRONT_RIGHT = None
        self.RADAR_BACK_LEFT = None
        self.RADAR_BACK_RIGHT = None
        self.LIDAR_TOP = None
        self.CAM_FRONT = None
        self.CAM_FRONT_RIGHT = None
        self.CAM_BACK_RIGHT = None
        self.CAM_BACK = None
        self.CAM_BACK_LEFT = None
        self.CAM_FRONT_LEFT = None

        self.update_sensor_calib_data()

    def update_sensor_calib_data(self):

        sensor_calib_tokens = [
            self.nusc.get('sample_data', self.current_sample['data'][sensor])['calibrated_sensor_token'] for sensor in
            self.sensors]
        self.sensor_calib_data = [self.nusc.get('calibrated_sensor', token) for token in sensor_calib_tokens]

        for idx, sensor in enumerate(self.sensors):

            # TODO: currently ignoring RADAR

            if sensor == 'LIDAR_TOP':
                self.LIDAR_TOP = self.sensor_calib_data[idx]
            if sensor == 'CAM_FRONT':
                self.CAM_FRONT = self.sensor_calib_data[idx]
            if sensor == 'CAM_FRONT_RIGHT':
                self.CAM_FRONT_RIGHT = self.sensor_calib_data[idx]
            if sensor == 'CAM_BACK_RIGHT':
                self.CAM_BACK_RIGHT = self.sensor_calib_data[idx]
            if sensor == 'CAM_BACK':
                self.CAM_BACK = self.sensor_calib_data[idx]
            if sensor == 'CAM_BACK_LEFT':
                self.CAM_BACK_LEFT = self.sensor_calib_data[idx]
            if sensor == 'CAM_FRONT_LEFT':
                self.CAM_FRONT_LEFT = self.sensor_calib_data[idx]

    def inverse_homegenous_transform(self, T_matrix):
        inv_T = np.eye(4)
        rot_inv = T_matrix[:3, :3].T
        trans = -T_matrix[:3, 3]
        inv_T[:3, :3] = rot_inv
        inv_T[:3, 3] = rot_inv.dot(trans)
        return inv_T

    def global_pose(self, sample, inverse=False):

        self.current_sample = sample
        # ego vehicle pose for each sensor (synced at the time of recording of respective data)
        ego_pose_tokens = [self.nusc.get('sample_data', sample['data'][sensor])['ego_pose_token'] for sensor in
                           self.sensors]
        ego_poses = [self.nusc.get('ego_pose', token) for token in ego_pose_tokens]

        num_sensors = len(self.sensor_calib_data)
        assert len(ego_poses) == num_sensors

        transform_matrices = []

        # list of 7x1 vectors Quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
        transform_vectors = []

        for idx in range(num_sensors):

            # transformation matrix from world frame to ego vehicle frame 
            T_vg = transform_matrix(translation=ego_poses[idx]['translation'],
                                    rotation=Quaternion(ego_poses[idx]['rotation']),
                                    inverse=True)

            # test sensor transform only 
            # T_vg = np.eye(4)

            # transformation matrix from ego vehicle frame to sensor frame
            T_sv = transform_matrix(translation=self.sensor_calib_data[idx]['translation'],
                                    rotation=Quaternion(self.sensor_calib_data[idx]['rotation']),
                                    inverse=True
                                    )

            # transformation matrix from global frame to sensor 
            T_sg = T_sv @ T_vg

            T_matrix = T_sg
            self.T_sg[self.sensors[idx]] = T_matrix

            if inverse:
                # transformation matrix from sensor frame to global frame
                T_matrix = self.inverse_homegenous_transform(T_matrix)
                self.T_gs[self.sensors[idx]] = T_matrix

            quaternion_vector = Quaternion(matrix=T_matrix[:3, :3]).elements

            # translation = -T_gs[:3, :3].T @ T_gs[:3, 3]
            translation = T_matrix[:3, 3]

            # append translation elements
            transform_vectors.append(np.append(quaternion_vector, translation.T))
            transform_matrices.append(T_matrix)

            self.update_sensor_calib_data()

        return transform_vectors, transform_matrices


class NovelCamera():
    def __init__(self, name, ref_extrinsics, ref_intrinsics):
        self.name = name
        self.ref_extrinsics = ref_extrinsics
        self.intrinsics = ref_intrinsics
        self.extrinsics = self.get_extrinsics()

    def get_extrinsics(self, inverse=False):
        T_cam_ref = transform_matrix(translation=self.ref_extrinsics['translation'],
                                     rotation=self.ref_extrinsics['rotation'],
                                     )
        T_ref_g = self.ref_extrinsics['T_ref']
        T_cam_g = T_cam_ref @ T_ref_g

        self.extrinsics = T_cam_g

        if inverse:
            T_g_cam = self.inverse_homegenous_transform(T_cam_g)
            return T_g_cam

        return T_cam_g

    @staticmethod
    def inverse_homegenous_transform(T_matrix):
        inv_T = np.eye(4)
        rot_inv = T_matrix[:3, :3].T
        trans = -T_matrix[:3, 3]
        inv_T[:3, :3] = rot_inv
        inv_T[:3, 3] = rot_inv.dot(trans)
        return inv_T

    def get_transform_vector(self):
        T_cam_g = self.extrinsics
        quaternion_vector = Quaternion(matrix=T_cam_g).elements
        translation = -T_cam_g[:3, :3].T @ T_cam_g[:3, 3]
        transform_vector = np.append(quaternion_vector, translation.T)
        return transform_vector


def get_novel_cam_params(sensor_params):
    rotations_plus = Quaternion(axis=[0.0, 1.0, 0.0], radians=np.pi / 6)
    rotation_minus = Quaternion(axis=[0.0, 1.0, 0.0], radians=-np.pi / 6)
    rotations = [rotations_plus, rotation_minus]
    novel_cameras = []

    sensors = sensor_params.sensors

    novel_cam_id = 1
    ref_intrinsics = sensor_params.CAM_FRONT_RIGHT['camera_intrinsic']

    for sensor in sensors:
        if sensor in ['CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
            ref_extrinsics = sensor_params.T_sg[sensor]
            for rotation in rotations:
                extrinsics = {
                    'T_ref': ref_extrinsics,  # 4x4 transformation matrix
                    'rotation': rotation,  # Unit quaternions
                    'translation': [0.0, 0.0, 0.0]
                }

                novel_camera = NovelCamera(name=f'NOVEL_CAM_{novel_cam_id}', ref_extrinsics=extrinsics,
                                           ref_intrinsics=ref_intrinsics)
                novel_cameras.append(novel_camera)
                novel_cam_id += 1

    return novel_cameras


def inverse_homegenous_transform(T_matrix):
    inv_T = np.eye(4)
    rot_inv = T_matrix[:3, :3].T
    trans = -T_matrix[:3, 3]
    inv_T[:3, :3] = rot_inv
    inv_T[:3, 3] = rot_inv.dot(trans)
    return inv_T


def coco_to_nuscenes(id, segment_info):
    coco_things_nuscenes_dict = {
        1: 2,  # bicycle
        5: 3,  # bus
        2: 4,  # car
        3: 6,  # motorcycle
        0: 7,  # pedestrian
        7: 10,  # truck
    }

    coco_stuffs_nuscenes_dict = {

        # barrier
        31: 1,  # wall-stone
        52: 1,  # barrier
        38: 1,  # fence

        21: 11,  # driveable surface

        # flat surface (playingfield, sand, gravel)
        18: 12,
        23: 12,
        11: 12,

        44: 13,  # sidewalk

        # terrain (mountain, grass, dirt)
        45: 14,
        46: 14,
        47: 14,

        # manmade (things, bridge, house, railroad, 
        # window, building)
        0: 15,
        3: 15,
        12: 15,
        19: 15,
        30: 15,
        32: 15,
        33: 15,
        36: 15,
        50: 15,

        # vegetation (tree, grass, flower)
        37: 16,
        9: 16
    }

    id_segment_info = segment_info[id - 1]
    if id_segment_info['isthing']:
        try:
            nuscene_seg_id = coco_things_nuscenes_dict[id_segment_info['category_id']]
        except:
            nuscene_seg_id = 0
    else:
        try:
            nuscene_seg_id = coco_stuffs_nuscenes_dict[id_segment_info['category_id']]
        except:
            nuscene_seg_id = 0

    return int(nuscene_seg_id)


def remap_nuscenes_lidarseg(lidarseg_old):
    lidarseg_new = np.zeros_like(lidarseg_old)
    remap_dict = {1: 0, 2: 7, 3: 7, 4: 7,
                  5: 0, 6: 7, 7: 0, 8: 0,
                  9: 1, 10: 0, 11: 0, 12: 8,
                  13: 0, 14: 2, 15: 3, 16: 3,
                  17: 4, 18: 5, 19: 9, 20: 0,
                  21: 6, 22: 9, 23: 10, 24: 11,
                  25: 12, 26: 13, 27: 14, 28: 15,
                  29: 0, 30: 16, 31: 0}

    max_value = max(max(remap_dict.values()), max(remap_dict.keys()))
    lookup_table = np.zeros(max_value + 1, dtype=lidarseg_old.dtype)

    for key, value in remap_dict.items():
        lookup_table[key] = value

    lidarseg_new = lookup_table[lidarseg_old]

    return lidarseg_new


def convert_to_int(lst):
    return [int(x) for x in lst]


def normalize_depth(image):
    array = np.array(image)

    min_value = np.min(array)
    max_value = np.max(array)

    normalized_array = (array - min_value) / (max_value - min_value)

    return normalized_array


def save_segmentation_iou(path, all_IOU, all_mIOU):
    all_mIOU_path = os.path.join(path, "mIOU.csv")
    with open(all_mIOU_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(all_mIOU)

    all_IOU_path = os.path.join(path, "mIOU.csv")
    with open(all_IOU_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(len(all_IOU)):
            for sublist in all_IOU[i]:
                writer.writerow(sublist)
