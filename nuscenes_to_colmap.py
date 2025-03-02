import os
import shutil
import argparse
import open3d as o3d
from nuscenes.nuscenes import NuScenes
import scripts.process_nuscenes as process_nuscenes
import scripts.colmap_utils as colmap_utils
from scripts.utils import SensorParameters, get_novel_cam_params

def parse_args():
    parser = argparse.ArgumentParser(description="Process NuScenes dataset for COLMAP and novel view synthesis.")
    parser.add_argument("--scene_idx", type=int, default=1, help="Scene index to process.")
    parser.add_argument("--set_size", type=int, default=5, help="Number of samples per set. Use <1 to take entire sequence as a single sample.")
    parser.add_argument("--samples_per_scene", type=int, default=1, help="Number of samples to process per scene.")
    parser.add_argument("--use_lidar", action="store_true", help="Use LiDAR data if specified.")
    parser.add_argument("--dataroot", type=str, default="data/sets/nuscenes", help="Path to NuScenes dataset root.")
    return parser.parse_args()

def setup_directories(directory_path, use_lidar):
    sub_dirs = ["sparse/0", "manual_sparse", "novel", "depth", "images"]
    if use_lidar:
        sub_dirs.append("lidar")
    
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(directory_path, sub_dir), exist_ok=True)

def process_scene(nusc, scene_idx, set_size, samples_per_scene, use_lidar, dataroot):
    directory_path = os.path.join(os.getcwd(), "data/colmap_data", f"scene-{scene_idx}")
    print(f"Created {directory_path}")

    os.makedirs(directory_path, exist_ok=True)
    
    scene = nusc.scene[scene_idx]
    scene_sample = nusc.get('sample', scene['first_sample_token'])
    cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    lidar = 'LIDAR_TOP' if use_lidar else None
    sensor_params = SensorParameters(nusc, scene_sample, sensors=cameras)
    
    # initialize variables
    sample_count, set_fill, img_id = 1, 0, 1
    
    print(f"Processing scene. name: {scene['name']} Token: {scene['token']}")
    while scene_sample['next'] != '' and sample_count <= samples_per_scene:
        sample_dir = os.path.join(directory_path, f"sample-{sample_count:02}")
        setup_directories(sample_dir, use_lidar)
        
        colmap_manual_sparse_folder = os.path.join(sample_dir, "manual_sparse")
        points3D_file = os.path.join(colmap_manual_sparse_folder, "points3D.txt")
        open(points3D_file, "w").close()  # Create empty points3D.txt file
        
        # Write camera intrinsics
        colmap_utils.write_intrinsics_file_nuscenes(colmap_manual_sparse_folder, sensor_params, img_width=1600, img_height=900)
        
        transform_vectors, _ = sensor_params.global_pose(scene_sample)
        novel_cam_params = get_novel_cam_params(sensor_params)
        novel_cam_intrinsics = [cam.intrinsics for cam in novel_cam_params]
        novel_cam_extrinsics = [cam.get_transform_vector() for cam in novel_cam_params]
        
        colmap_utils.write_intrinsics_file_novelcam(os.path.join(sample_dir, "novel"), novel_cam_intrinsics)
        
        save_depth_paths = []
        
        # Copy LiDAR data if required
        if use_lidar:
            lidar_token = scene_sample['data'][lidar]
            lidar_data = nusc.get('sample_data', lidar_token)
            pcl_path = os.path.join(dataroot, nusc.get('sample_data', lidar_token)['filename'])
            pc = process_nuscenes.project_pointcloud_global_frame(nusc, pcl_path, lidar_data)
            pc_numpy = pc.points.T[:, :3]
            o3d_pcl = o3d.geometry.PointCloud()
            o3d_pcl.points = o3d.utility.Vector3dVector(pc_numpy)
            o3d.io.write_point_cloud(os.path.join(sample_dir, "lidar", f"lidar-{sample_count:02}.pcd"), o3d_pcl)
        
        # Copy images and write extrinsics
        with open(os.path.join(colmap_manual_sparse_folder, "images.txt"), "w") as file:
            for idx, tv in enumerate(transform_vectors):
                filename = nusc.get('sample_data', scene_sample['data'][cameras[idx]])['filename']
                source_path = os.path.join(dataroot, filename)
                target_path = os.path.join(sample_dir, "images", f"image-{sample_count:02}-{img_id:02}.jpg")
                shutil.copy(source_path, target_path)
                file.write(f"{img_id} {tv[0]} {tv[1]} {tv[2]} {tv[3]} {tv[4]} {tv[5]} {tv[6]} {idx + 1} image-{sample_count:02}-{img_id:02}.jpg\n\n")
                save_depth_paths.append(os.path.join(sample_dir, "depth", f"image-{sample_count:02}-{img_id:02}.png"))
                img_id += 1
        
        process_nuscenes.img_lidar_depth_map(nusc, scene_sample, cameras, plot_img=False, save_paths=save_depth_paths)
        print(f"Processed sample {sample_count}, set {set_fill}")

        scene_sample = nusc.get('sample', scene_sample['next'])
        
        set_fill += 1
        if set_size > 1 and set_fill % set_size == 0:
            sample_count += 1
            set_fill = 0
            img_id = 1
        
        colmap_utils.write_batch_file(sample_dir, colmap_manual_sparse_folder, os.path.join(sample_dir, "sparse/0"))
    
    # Delete Temp file
    colmap_utils.write_intrinsics_file_nuscenes(delete_temp=True)
    print(f"Finished processing scene {scene['name']}")

def main():
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    process_scene(nusc, args.scene_idx, args.set_size, args.samples_per_scene, args.use_lidar, args.dataroot)

if __name__ == "__main__":
    main()