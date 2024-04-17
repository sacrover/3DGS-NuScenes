from nuscenes.nuscenes import NuScenes
import os

import scripts.process_nuscenes as process_nuscenes
import scripts.colmap_utils as colmap_utils
from scripts.utils import SensorParameters, get_novel_cam_params

import shutil


def main():
    # TODO argparse, tqdm

    set_size = 5  # set to < 1 to take entire sequence as a single sample
    scene_idx = 1
    directory_path = os.path.join(os.getcwd(), "data/colmap_data", f"scene-{scene_idx}")
    samples_per_scene = 1 

    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
    cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    scene = nusc.scene[scene_idx]
    first_sample_token = scene['first_sample_token']
    scene_sample = nusc.get('sample', first_sample_token)
    sensor_params = SensorParameters(nusc, scene_sample, sensors=cameras)
    scene_sample = nusc.get('sample', first_sample_token)

    sample_count = 1
    set_fill = 0
    img_id = 1

    while scene_sample['next'] != '' and sample_count <= samples_per_scene:
        sample_dir = os.path.join(directory_path, f"sample-{sample_count:02}")
        images_dir = os.path.join(sample_dir, "images")

        colmap_sparse_folder = os.path.join(sample_dir, "sparse/0")
        colmap_manual_sparse_folder = os.path.join(sample_dir, "manual_sparse")
        novel_camera_folder = os.path.join(sample_dir, "novel")
        points3D_file = os.path.join(colmap_manual_sparse_folder, "points3D.txt")

        depth_map_folder = os.path.join(sample_dir, "depth")

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if not os.path.exists(colmap_manual_sparse_folder):
            os.makedirs(colmap_manual_sparse_folder)

        if not os.path.exists(novel_camera_folder):
            os.makedirs(novel_camera_folder)

        if not os.path.exists(colmap_sparse_folder):
            os.makedirs(colmap_sparse_folder)

        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        if not os.path.exists(depth_map_folder):
            os.makedirs(depth_map_folder)

        with open(points3D_file, "w") as file:
            pass

        colmap_utils.write_intrinsics_file_nuscenes(colmap_manual_sparse_folder, sensor_params, img_width=1600,
                                                    img_height=900)

        colmap_extrinsics_file_path = os.path.join(colmap_manual_sparse_folder, "images.txt")
        transform_vectors, _ = sensor_params.global_pose(scene_sample)

        novel_cam_params = get_novel_cam_params(sensor_params)
        novel_cam_intrinsics = [cam.intrinsics for cam in novel_cam_params]
        novel_cam_extrinsics = [cam.get_transform_vector() for cam in novel_cam_params]

        colmap_utils.write_intrinsics_file_novelcam(novel_camera_folder, novel_cam_intrinsics)

        mode = "w"

        if set_fill != 0:
            mode = "a"

        colmap_utils.write_extrinsics_file_novelcam(novel_camera_folder, mode, novel_cam_extrinsics)

        save_depth_paths = []

        # Write Extrinsics
        with open(colmap_extrinsics_file_path, mode) as file:
            for idx, tv in enumerate(transform_vectors):
                filename = nusc.get('sample_data', scene_sample['data'][cameras[idx]])['filename']
                source_path = os.path.join("data/sets/nuscenes", filename)
                target_path = os.path.join(images_dir, f"image-{sample_count:02}-{img_id:02}.jpg")
                shutil.copy(source_path, target_path)
                line = f"{img_id} {tv[0]} {tv[1]} {tv[2]} {tv[3]} {tv[4]} {tv[5]} {tv[6]} {idx + 1} image-{sample_count:02}-{img_id:02}.jpg\n"
                file.write(line)
                file.write("\n")  # Write an empty line

                save_depth_paths.append(os.path.join(depth_map_folder, f"image-{sample_count:02}-{img_id:02}.png"))
                img_id += 1

        process_nuscenes.img_lidar_depth_map(nusc, scene_sample, cameras, plot_img=False, save_paths=save_depth_paths)

        scene_sample = nusc.get('sample', scene_sample['next'])
        set_fill += 1

        if set_size > 1:
            if set_fill % set_size == 0:
                sample_count += 1
                set_fill = 0
                img_id = 1

        colmap_utils.write_batch_file(sample_dir, colmap_manual_sparse_folder, colmap_sparse_folder)

    # Delete Temp file
    colmap_utils.write_intrinsics_file_nuscenes(delete_temp=True)


if __name__ == "__main__":
    main()
