import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

from matplotlib import pyplot as plt


class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load mode config and pretrained model 

        # Panoptic FPN R 101 Cascade GN performed better for the NuScenes dataset 
        self.cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml")

        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS_SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

        self.coco_metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.stuff_classes = self.coco_metadata.stuff_classes
        self.thing_classes = self.coco_metadata.thing_classes

    def onImage(self, image_path, plot=False):
        image = cv2.imread(image_path)
        predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

        viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

        output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        output_image = cv2.cvtColor(output.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

        if plot:
            plt.imshow(output_image)
            plt.show()

        return predictions.cpu().detach().numpy(), segmentInfo

    def print_metadata(self):
        print("STUFFS CLASSES:\n")
        for idx, category_name in enumerate(self.stuff_classes):
            print(f"{idx}: {category_name}")

        print("\nTHINGS CLASSES\n")
        for idx, category_name in enumerate(self.thing_classes):
            print(f"{idx}: {category_name}")


def main():
    return


if __name__ == "__main__":
    main()
