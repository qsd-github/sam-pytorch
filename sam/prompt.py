import os
import cv2 as cv
import numpy as np
import torch
from segment_anything import sam_model_registry

from sam.base import SAMBase
from sam.utils.display.matplotlib import Matplotlib
from sam.utils.display.opencv import OpenCV
from sam.utils.url.weights import Weight

project_path = os.path.dirname(__file__)
checkpoint_path = os.path.join(project_path, "./sam_vit_b.pth")


class SAMPrompt(SAMBase):
    def __init__(self, image_path, input_points=None,
                 input_labels=None, input_boxes=None, optim_masks=None, text=None, model_type="vit_b", sam_checkpoint_path=None, is_multimask=True, is_cuda=True):
        self._image_path = image_path
        self._image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

        self._input_points = input_points
        self._input_labels = input_labels
        self._input_boxes = input_boxes
        self._optim_masks = optim_masks
        self._text = text
        self._is_multimask = is_multimask

        model_url = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        self._model_type = model_type
        self._sam_checkpoint_path = sam_checkpoint_path
        self._is_cuda = is_cuda

        if sam_checkpoint_path is None:
            weight = os.listdir(os.path.join(project_path, "./"))
            if "sam_" + model_type + ".pth" not in weight:
                Weight.download(model_type, model_url[model_type],
                                os.path.join(project_path, "./", "sam_" + model_type + ".pth"))
            self.__sam_checkpoint_path = os.path.join(project_path, "./", "sam_" + model_type + ".pth")

        if self._is_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                self._sam_model = sam_model_registry[model_type](checkpoint=self._sam_checkpoint_path).to(
                    device=device)
            else:
                raise RuntimeError("No GPU found on this computer!")
        else:
            self._sam_model = sam_model_registry[model_type](checkpoint=self._sam_checkpoint_path)

        self._mat_show = Matplotlib()
        self._cv_show = OpenCV()

    def image(self, type="matplotlib", is_save=False, save_path=None):
        file_name = os.path.basename(self._image_path)
        prompts_image = self._prompt_image()
        if type == "matplotlib":
            self._mat_show.show(prompts_image, file_name)
        elif type == "opencv":
            image = cv.cvtColor(prompts_image, cv.COLOR_RGB2BGR)
            self._cv_show.show(image, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
        if is_save:
            if save_path is None:
                raise ValueError("Please provide parameters: save_path!")
            image = cv.cvtColor(prompts_image, cv.COLOR_RGB2BGR)
            cv.imwrite(save_path, image)

    def segment(self):
        pass

    def predict(self, type="matplotlib", is_save=False, save_path=None):
        file_name = os.path.basename(self._image_path)
        masks, scores, _ = self.segment()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_image = self._get_mask_image(mask)
            if type == "matplotlib":
                self._mat_show.show(mask_image, f"{file_name} Mask {i + 1}, Score: {score.item():.3f}")
            elif type == "opencv":
                mask_image = cv.cvtColor(mask_image, cv.COLOR_RGB2BGR)
                self._cv_show.show(mask_image, file_name)
            else:
                raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']!")
            if is_save:
                if save_path is None:
                    raise ValueError("When is_save selects True, save_path cannot be None!")
                save_path_list = os.path.abspath(save_path).split(".")
                save_path_f = save_path_list[0] + str(i) + ".jpg"
                cv.imwrite(save_path_f, mask_image * 255)

    def save(self, save_path):
        masks, scores, _ = self.segment()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_image = self._get_mask_image(mask)
            save_path_list = os.path.abspath(save_path).split(".")
            save_path_f = save_path_list[0] + str(i) + ".jpg"
            cv.imwrite(save_path_f, mask_image * 255)

    def _prompt_image(self):
        pass

    def _get_mask_image(self, masks):
        color = np.random.random(3)
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def __getitem__(self, item):
        file_name = os.path.basename(self._image_path)
        masks, scores, _ = self.segment()
        if item > len(masks):
            raise IndexError("The index value should be less than the number of masks!")
        mask_image = self._get_mask_image(masks[item])
        self._mat_show.show(mask_image, file_name + " prompt masks")

