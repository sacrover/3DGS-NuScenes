from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud

import numpy as np
from scripts.panoptic_segmentation import Detector
import pandas as pd
import os
from PIL import Image

from scripts.plot_utils import plot_fig_legend
from scripts.plot_utils import plot_lidar_projection
from scripts.plot_utils import plot_depth_map

from pyquaternion import Quaternion

from scripts.utils import inverse_homegenous_transform
from scripts.utils import coco_to_nuscenes
from scripts.utils import convert_to_int

from scripts.metrics import compute_iou, mse


def project_pointcloud_nuscenes_default(nusc, pcl_path, cam_data, lidar_data):
    """
    TODO: Description
    """
    pc = LidarPointCloud.from_file(pcl_path)

    # transformation matrix from sensor frame to ego vehicle frame
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    T_sv = transform_matrix(translation=cs_record['translation'],
                            rotation=Quaternion(cs_record['rotation']),
                            inverse=True)

    # Second step: transform from ego vehicle to global frame.
    poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    T_vg = transform_matrix(translation=poserecord['translation'],
                            rotation=Quaternion(poserecord['rotation']),
                            inverse=True)
    
    pc.transform(inverse_homegenous_transform(T_sv @ T_vg))

    # transform from global to ego vehicle frame (based on camera timestamp) 
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    T_sv = transform_matrix(translation=cs_record['translation'],
                            rotation= Quaternion(cs_record['rotation']),
                            inverse=True
                            )
    
    # transform from ego vehicle frame to camera frame
    poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
    T_vg = transform_matrix(translation=poserecord['translation'],
                            rotation= Quaternion(poserecord['rotation']),
                            inverse=True)

    pc.transform(T_sv @ T_vg)

    depths = pc.points[2, :]

    points_cam_3d = pc.points
    nbr_points = points_cam_3d.shape[1]
    
    K = np.array(cs_record['camera_intrinsic'])
    viewpad = np.eye(4)
    viewpad[:K.shape[0], :K.shape[1]] = K

    points_cam_proj = np.dot(viewpad, points_cam_3d)
    points_cam_proj = points_cam_proj[:3, :]

    epsilon = 1e-6  # Small epsilon value

    # Add epsilon to the denominator to prevent division by zero
    denominator = points_cam_proj[2:3, :].repeat(3, 0).reshape(3, nbr_points) + epsilon

    points_cam_proj = points_cam_proj / denominator
    points_cam_proj = points_cam_proj.astype(int)

    return points_cam_proj, depths

def project_pointcloud_novel_cam(nusc, pcl_path, T_sg, intrinsics, lidar_data):
    """
    TODO: Description
    """
     
    pc = LidarPointCloud.from_file(pcl_path)
    
    # transformation matrix from sensor frame to ego vehicle frame
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    T_sv = transform_matrix(translation=cs_record['translation'],
                            rotation=Quaternion(cs_record['rotation']),
                            inverse=True)

    # Second step: transform from ego vehicle to global frame.
    poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    T_vg = transform_matrix(translation=poserecord['translation'],
                            rotation=Quaternion(poserecord['rotation']),
                            inverse=True)
    
    pc.transform(inverse_homegenous_transform(T_sv @ T_vg))

    pc.transform(T_sg)

    depths = pc.points[2, :]

    points_cam_3d = pc.points
    nbr_points = points_cam_3d.shape[1]
    print("num points: ", nbr_points)

    K = np.array(intrinsics)
    viewpad = np.eye(4)
    viewpad[:K.shape[0], :K.shape[1]] = K

    points_cam_proj = np.dot(viewpad, points_cam_3d)
    points_cam_proj = points_cam_proj[:3, :]

    epsilon = 1e-6  # Small epsilon value

    # Add epsilon to the denominator to prevent division by zero
    denominator = points_cam_proj[2:3, :].repeat(3, 0).reshape(3, nbr_points) + epsilon

    points_cam_proj = points_cam_proj / denominator
    points_cam_proj = points_cam_proj.astype(int)

    return points_cam_proj, depths

def lidar_pre_masks(points_cam_proj, depths, img_size, min_dist=1.0):
    pcl_mask = np.ones(depths.shape[0], dtype=bool)
    pcl_mask = np.logical_and(pcl_mask, depths > min_dist)
    pcl_mask = np.logical_and(pcl_mask, points_cam_proj[0, :] > 1)
    pcl_mask = np.logical_and(pcl_mask, points_cam_proj[0, :] < img_size[0] - 1)
    pcl_mask = np.logical_and(pcl_mask, points_cam_proj[1, :] > 1)
    pcl_mask = np.logical_and(pcl_mask, points_cam_proj[1, :] < img_size[1] - 1)

    return pcl_mask


def annotate_projected_lidar_img(detector, img_path, filtered, verbose=False):
    predictions, segment_info = detector.onImage(img_path)
    lidar_annotated = np.zeros(filtered.shape[1])
    for idx in range(filtered.shape[1]):
        lidar_annotated[idx] = coco_to_nuscenes(predictions[filtered[1, idx], filtered[0, idx]], segment_info)
    
    if verbose:
        print("Predictions\n", predictions, "\nNum Preds: ", np.max(predictions))
        print("\nSegment Info\n", segment_info)

    return lidar_annotated

def extract_unique_label(unique_vals):
    if len(unique_vals) == 1:
        # If only one unique value, return that value
        return unique_vals[0]
    elif len(unique_vals) == 0:
        # If all values are -1, return 17 (unassigned), can be -1 too
        return 17
    else:
        # If multiple unique values, return the one with max occurrence
        return max(unique_vals, key=unique_vals.count)
    
def annotate_pcl(pcl_masks, pcl_seg_classes):
    num_masks = len(pcl_masks)
    lidar_seg_raw = np.ones([len(pcl_masks[0]), num_masks]) * -1

    for i in range(num_masks):
        lidar_seg_raw[pcl_masks[i], i] =  pcl_seg_classes[i]

    df = pd.DataFrame(lidar_seg_raw)

    # Find the number of non-void assignments per row
    assign_counts = (df != -1).sum(axis=1)
    void_counts = (df == 0).sum(axis=1)

    # Find rows with more than one non-void assignment
    rows_with_multiple_assignment = df[assign_counts > 1].index

    # Find rows with no assignment
    rows_with_no_assignment = df[assign_counts == 0].index

    # Find rows with void assignments
    rows_with_all_void = df[void_counts >= 1].index

    print("Points with more than one assignments: ", len(rows_with_multiple_assignment))
    # print(rows_with_multiple_assignment.tolist()[:5])

    print("Points with void assignments: ", len(rows_with_all_void))
    # print(rows_with_all_void.tolist()[-10:], "...", )

    print("Points with no assignments: ",  len(rows_with_no_assignment))
    # print(rows_with_no_assignment.tolist()[-10:])

    # Replace -1 with NaN for better handling
    df_nan = df.replace(-1, np.nan)

    # Find unique non-NaN values along each row
    unique_values = df_nan.apply(lambda row: convert_to_int(row.dropna().unique()), axis=1)

    # Apply the function to each row to get unique labels
    unique_labels = unique_values.apply(extract_unique_label)

    return unique_labels

def lidar_segmentation(nusc, sample, plot_img=False, **kwargs):

    detector = Detector()
     
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    if plot_img:
        num_classes = 17 # nuScenes segmentation class 
        extra_plot_img_rows = 0
        if 'novel_cams' in kwargs and 'train_dir' in kwargs and 'novel_dir' in kwargs:
            extra_plot_img_rows = np.ceil(len(kwargs['novel_cams']) / 2).astype(int)
        
        fig, axs = plt.subplots(3 + extra_plot_img_rows, 2, figsize=(12, 12 + 2*extra_plot_img_rows))

        # Define a color map for the label classes
        class_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'teal', 'gray', 'pink', 'lightblue', 'lightgreen', 'lightgray', 'darkblue', 'brown','darkgreen']
        cmap = ListedColormap(class_colors)
        color_cycle = [cmap(i) for i in np.linspace(0, 1, num_classes)]

    cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    all_pcl_mask = []
    all_pcl_seg = []

    for i, cam in enumerate(cameras):
        camera_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', camera_token)
        
        if 'train_dir' in kwargs:
            img_path = os.path.join(kwargs['train_dir'], f"{(i + kwargs['start_img_num']):05d}.png")
        else:
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        
        img = Image.open(img_path)
        img_size = [img.width, img.height]

        points_cam_proj, depths = project_pointcloud_nuscenes_default(nusc, pcl_path, cam_data, lidar_data)

        img_size = [img.width, img.height]
        pcl_mask = lidar_pre_masks(points_cam_proj, depths, img_size)
        filtered = points_cam_proj[:, pcl_mask]
        annotated_img = annotate_projected_lidar_img(detector, img_path, filtered, verbose=False)
        plot_segmented = True

        if plot_img:
            ax = axs[i // 2, i % 2]  # Calculate subplot index
            plot_lidar_projection(ax, img, plot_segmented, color_cycle, filtered, annotated_img)
            ax.set_title(cam)

        all_pcl_mask.append(pcl_mask)
        all_pcl_seg.append(annotated_img)

    if 'novel_cams' in kwargs and 'novel_dir' in kwargs:
        for j, novel_cam in enumerate(kwargs['novel_cams']):
            name = novel_cam.name
            novel_cam_id = int(name[name.rfind('_') + 1:])
            img_path = os.path.join(kwargs['novel_dir'], f"{(novel_cam_id-1):05d}.png")
            img = Image.open(img_path)
            img_size = [img.width, img.height]
            T_sg = novel_cam.get_extrinsics()
            intrinsics = novel_cam.intrinsics
            points_cam_proj, depths = project_pointcloud_novel_cam(nusc, pcl_path, T_sg, intrinsics, lidar_data)

            img_size = [img.width, img.height]
            pcl_mask = lidar_pre_masks(points_cam_proj, depths, img_size)
            filtered = points_cam_proj[:, pcl_mask]
            lidar_segmented = annotate_projected_lidar_img(detector, img_path, filtered, verbose=False)

            if plot_img:
                ax = axs[(j + 1 + i)// 2, j % 2]  # Calculate subplot index
                plot_lidar_projection(ax, img, plot_segmented, color_cycle, filtered, lidar_segmented)
                ax.set_title(name)

            all_pcl_mask.append(pcl_mask)
            all_pcl_seg.append(lidar_segmented)

    if plot_img:
        plot_fig_legend(fig, color_cycle)
        plt.show()

    return all_pcl_mask, pcl_path, all_pcl_seg

def img_lidar_depth_map(nusc, sample, cameras, plot_img=False, save_paths=None):
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    fig = None
    if plot_img:
        fig, axs = plt.subplots(3, 2, figsize=(15,12))

    for i, cam in enumerate(cameras):
        camera_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', camera_token)
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        img = Image.open(img_path)
        img_size = [img.width, img.height]

        points_cam_proj, depths = project_pointcloud_nuscenes_default(nusc, pcl_path, cam_data, lidar_data)

        img_size = [img.width, img.height]
        pcl_mask = lidar_pre_masks(points_cam_proj, depths, img_size)
        filtered_points = points_cam_proj[:, pcl_mask].T
        filtered_depths = depths[pcl_mask]


        if plot_img or save_paths is not None:
            
            ax = None
            save_path = None

            # plot_lidar_projection(ax, img, filtered)
            if plot_img: 
                ax = axs[i // 2, i % 2]  # Calculate subplot index

            if save_paths:
                save_path = save_paths[i]

            plot_depth_map(img.width, img.height, filtered_points, filtered_depths, plot=plot_img, ax=ax, fig=fig, save_path=save_path)

            if plot_img: 
                ax.set_title(cam)

    if plot_img:
        plt.show()

def compute_depth_mse(scene_idxs, root_dirs=None):
    for scene_idx in scene_idxs:
        depth_mse = 0
        images_dir = next((d for d in os.listdir(root_dirs[scene_idx]) if d.startswith("ours_")), None)
        print(images_dir)
        render_images_dir = os.path.join(root_dirs[scene_idx], images_dir, "renders")
        gt_images_dir = os.path.join(root_dirs[scene_idx], images_dir, "gt" )

        # List the files in both directories
        files1 = sorted(os.listdir(render_images_dir))
        files2 = sorted(os.listdir(gt_images_dir))

        # Iterate through the files in parallel
        for file1, file2 in zip(files1, files2):
            # Open the images
            render_image = Image.open(os.path.join(render_images_dir, file1))
            gt_image = Image.open(os.path.join(gt_images_dir, file2))

            depth_mse += mse(render_image, gt_image)

        print(f"Depth MSE for scene {scene_idx}: ", depth_mse/len(files1))

def main():
    # TODO: arg parser to select scene id, sample skip

    scene_idx = 0
    skip_keyframe = 0
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
    scene = nusc.scene[scene_idx]

    first_sample_token = scene['first_sample_token']

    first_sample_token = scene['first_sample_token']
    scene_sample = nusc.get('sample', first_sample_token)

    for _ in range(skip_keyframe):
        scene_sample = nusc.get('sample', scene_sample['next'])

    pcl_masks_nusc, pcl_path, pcl_seg_classes_nusc = lidar_segmentation(nusc, scene_sample, plot_img=True)