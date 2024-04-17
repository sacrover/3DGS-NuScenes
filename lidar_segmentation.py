import numpy as np
from nuscenes.nuscenes import NuScenes
import os
import scripts.process_nuscenes as process_nuscenes
from scripts.utils import remap_nuscenes_lidarseg
from scripts.metrics import compute_iou
from scripts.utils import save_segmentation_iou


def compute_batch_iou(nusc: NuScenes, scene_idxs, img_dir_paths=None):
    all_IOU = []
    all_mIOU = []
    for scene_idx in scene_idxs:
        scene = nusc.scene[scene_idx]
        first_sample_token = scene['first_sample_token']
        scene_sample = nusc.get('sample', first_sample_token)
        frame = 0
        all_frame_IOU = []
        all_frame_mIOU = []
        while frame < 5:
            print(f"\nSCENE ID: {scene_idx}, FRAME: {frame} \n")

            lidar_token = scene_sample['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
            gt_seg_classes_old = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
            gt_seg_classes_new = remap_nuscenes_lidarseg(gt_seg_classes_old)

            root_dir = img_dir_paths[scene_idx]

            # Find the directory starting with "ours_"
            images_dir = next((d for d in os.listdir(root_dir) if d.startswith("ours_")), None)

            images_dir = os.path.join(root_dir, images_dir, "gt")

            pcl_masks, pcl_path, pcl_seg_classes = process_nuscenes.lidar_segmentation(nusc, scene_sample,
                                                                                       plot_img=False,
                                                                                       novel_cams=None,
                                                                                       train_dir=images_dir,
                                                                                       novel_dir=None,
                                                                                       start_img_num=frame * 6)

            model_label = process_nuscenes.annotate_pcl(pcl_masks=pcl_masks, pcl_seg_classes=pcl_seg_classes)
            IoU, mIOU = compute_iou(model_label, gt_seg_classes_new)
            all_frame_IOU.append(IoU)
            all_frame_mIOU.append(mIOU)

            scene_sample = nusc.get('sample', scene_sample['next'])
            frame += 1

        all_IOU.append(all_frame_IOU)
        all_mIOU.append(all_frame_mIOU)

    return all_IOU, all_mIOU


def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=False)
    scene_idxs = [0, 1, 2, 3, 4]
    sample_id = 1
    depth_reg_paths = [f"test-scn{scene_idx}-set5-smp{sample_id:02d}-depth/train" for scene_idx in scene_idxs]
    no_depth_reg_paths = [f"test-scn{scene_idx}-set5-smp01-nodepth/train" for scene_idx in scene_idxs]
    gt_paths = [f"test-scn{scene_idx}-set5-smp01-depth/train" for scene_idx in scene_idxs]

    for paths in [depth_reg_paths, no_depth_reg_paths, gt_paths]:
        all_IOU, all_mIOU = compute_batch_iou(nusc, scene_idxs, img_dir_paths=paths)
        save_path = f"../test-scn{scene_idxs[0]}-set5-smp01-depth/train"
        save_segmentation_iou(path=save_path, all_IOU=all_IOU, all_mIOU=all_mIOU)


if __name__ == "__main__":
    main()
