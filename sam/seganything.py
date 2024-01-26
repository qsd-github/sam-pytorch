import os.path

from sam.base import SAMBase
import cv2 as cv
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from sam.utils.display.matplotlib import Matplotlib
from sam.utils.display.opencv import OpenCV
from sam.utils.url.weights import Weight

project_path = os.path.dirname(__file__)
checkpoint_path = os.path.join(project_path, "./sam_vit_b.pth")


class SAMAny(SAMBase):
    def __init__(self, image_path, model_type="vit_b", sam_checkpoint_path=None, is_cuda=True):
        """
        :param image_path:the address where the image needs to be segmented
        :param model_type:Category of SAM model
        :param sam_checkpoint_path:SAM weight path, if not provided, it will be automatically downloaded
        :param is_cuda:is or is not GPU acceleration used
        """
        self.__image_path = image_path
        self.__image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

        model_url = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        self.__model_type = model_type
        self.__sam_checkpoint_path = sam_checkpoint_path
        self.__is_cuda = is_cuda

        if sam_checkpoint_path is None:
            weight = os.listdir(os.path.join(project_path, "./"))
            if "sam_" + model_type + ".pth" not in weight:
                Weight.download(model_type, model_url[model_type], os.path.join(project_path, "./", "sam_" + model_type + ".pth"))
            self.__sam_checkpoint_path = os.path.join(project_path, "./", "sam_" + model_type + ".pth")

        if self.__is_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                self.__sam_model = sam_model_registry[model_type](checkpoint=self.__sam_checkpoint_path).to(device=device)
            else:
                raise RuntimeError("No GPU found on this computer!")
        else:
            self.__sam_model = sam_model_registry[model_type](checkpoint=self.__sam_checkpoint_path)

        self.__mat_show = Matplotlib()
        self.__cv_show = OpenCV()

    def image(self, type="matplotlib", is_save=None, save_path=None):
        """
        This method is used to display the original input image
        :param type: You can choose between "matplotlib" or "opencv" these two types
        """
        file_name = os.path.basename(self.__image_path)
        if type == "matplotlib":
            self.__mat_show.show(self.__image, file_name)
        elif type == "opencv":
            image = cv.cvtColor(self.__image, cv.COLOR_RGB2BGR)
            self.__cv_show.show(image, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")

    def segment(self):
        """
        segment all masks, a single image may have multiple masks
        :return "all"<dic>: return all masks\n
        """
        mask_generator = SamAutomaticMaskGenerator(self.__sam_model)
        masks = mask_generator.generate(self.__image)
        return masks

    def predict(self, type="matplotlib", is_save=False, save_path=None):
        """
        predictive segmentation of images
        :param type:You can choose between "matplotlib" or "opencv" these two types
        :param is_save:save or not save the predicted image
        :param save_path:save the position of the predicted image, if is_save is true, fill in this parameter
        """
        file_name = os.path.basename(self.__image_path)
        masks = self.segment()
        mask_image = self.__get_mask_image(masks)
        if type == "matplotlib":
            self.__mat_show.show(mask_image, file_name + " all masks")
        elif type == "opencv":
            self.__cv_show.show(mask_image, file_name + " all masks")
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
        if is_save:
            if save_path is None:
                raise ValueError("Please provide parameters: save_path!")
            cv.imwrite(save_path, mask_image * 255)

    def save(self, save_path):
        """
        save the position of the predicted image
        :param save_path:save image location
        """
        masks = self.segment()
        mask_image = self.__get_mask_image(masks)
        cv.imwrite(save_path, mask_image * 255)

    def __get_mask_image(self, masks):
        """
        Display mask images
        :param masks: segment anything masks
        :return: mask image
        """
        if len(masks) == 0:
            raise ValueError("Masks is empty!")
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random(3)
            img[m] = color_mask
        return img

    def segmentations(self):
        """
        segmentation: <numpy.array>
        :return:the segmented binary graph
        """
        masks = self.segment()
        return [mask['segmentation'].astype(np.uint8) * 255 for mask in masks]

    def areas(self):
        """
        area: <int>
        :return:the area of the mask.
        """
        masks = self.segment()
        return [mask['area'] for mask in masks]

    def boxes(self):
        """
        "bbox": <list>
        :return:the box used for segmenting objects
        """
        masks = self.segment()
        return [mask['bbox'] for mask in masks]

    def predicted_ious(self):
        """
        "predicted_iou": float
        :return: mask prediction quality
        """
        masks = self.segment()
        return [mask['predicted_iou'] for mask in masks]

    def point_coords(self):
        """
        "point_coords": list
        :return:the sampling output point that generated this mask
        """
        masks = self.segment()
        return [mask['point_coords'] for mask in masks]

    def stability_scores(self):
        """
        "stability_score": float
        :return:stability score, another evaluation parameter for mask quality
        """
        masks = self.segment()
        return [mask['stability_score'] for mask in masks]

    def crop_boxes(self):
        """
        "crop_box": list
        :return:Mask image clipping, return format is [X, Y, W, H]
        """
        masks = self.segment()
        return [mask['crop_box'] for mask in masks]