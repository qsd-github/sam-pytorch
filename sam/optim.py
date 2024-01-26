from typing import Union

from segment_anything import SamPredictor
import os
import cv2 as cv
import numpy as np
import torch
from segment_anything import sam_model_registry

from sam.utils.display.matplotlib import Matplotlib
from sam.utils.display.opencv import OpenCV
from sam.utils.url.weights import Weight
from sam.prompt import SAMPrompt

project_path = os.path.dirname(__file__)
checkpoint_path = os.path.join(project_path, "./sam_vit_b.pth")


class SAMOptim(SAMPrompt):
    def __init__(self, image: Union[str, np.ndarray], optim_masks, model_type="vit_b", sam_checkpoint_path=None, is_multimask=True, is_cuda=True):
        if isinstance(image, str):
            self._image_path = image
            self._image = cv.cvtColor(cv.imread(self._image_path), cv.COLOR_BGR2RGB)
            self.H, self.W = self._image.shape[0], self._image.shape[1]
            self._image = cv.resize(self._image, (256, 256))
        elif isinstance(image, np.ndarray):
            self._image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            self.H, self.W = self._image.shape[0], self._image.shape[1]
            self._image = cv.resize(self._image, (256, 256))

        self._optim_masks = optim_masks
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
        if type == "matplotlib":
            self._mat_show.connect_show(cv.resize(self._image.copy(), (self.W, self.H)), self._optim_masks, f"{file_name} origin image", f"{file_name} mask image")
        elif type == "opencv":
            # image = cv.cvtColor(self._optim_masks, cv.COLOR_RGB2BGR)
            self._cv_show.connect_show(cv.resize(self._image.copy(), (self.W, self.H)), self._optim_masks, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
        if is_save:
            if save_path is None:
                raise ValueError("Please provide parameters: save_path!")
            image = cv.cvtColor(self._optim_masks, cv.COLOR_RGB2BGR)
            cv.imwrite(save_path, image)


    def segment(self):
        predictor = SamPredictor(self._sam_model)
        predictor.set_image(self._image)

        masks, scores, logits = predictor.predict(
            # mask_input=np.resize(self._optim_masks, (1, 256, 256)),  # 输入为掩码
            mask_input=np.resize(self._image[:, :, 0:1].transpose((2, 0, 1)), (1, 256, 256)),  # 输入为掩码
            multimask_output=False  # 单mask输出，选择scores值最高的。
        )
        return masks, scores, logits

    def predict(self, type="matplotlib", is_save=False, save_path=None):
        file_name = os.path.basename(self._image_path)
        masks, scores, _ = self.segment()
        mask_image = self._get_mask_image(masks)
        if type == "matplotlib":
            self._mat_show.show(mask_image, f"{file_name} optimize")
        elif type == "opencv":
            mask_image = cv.cvtColor(mask_image, cv.COLOR_RGB2BGR)
            self._cv_show.show(mask_image, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']!")
        if is_save:
            if save_path is None:
                raise ValueError("When is_save selects True, save_path cannot be None!")
            cv.imwrite(save_path, mask_image)

    def save(self, save_path):
        masks, scores, _ = self.segment()
        mask_image = self._get_mask_image(masks)
        cv.imwrite(save_path, mask_image)

    def _prompt_image(self):
        if self._optim_masks.shape[0] != 1:
            raise ValueError("please keep the number of channels to 1, such as (1, 224, 224)!")
        fusion_image = self._image.copy()
        fusion_image[self._optim_masks[0] != 0] = (255, 0, 0)
        return fusion_image

    def _get_mask_image(self, masks):
        masks = masks.astype(np.uint8)
        masks[masks == 1] = 255
        masks = cv.resize(masks.transpose((1, 2, 0)), (self.W, self.H))
        return masks

    def __getitem__(self, item):
        file_name = os.path.basename(self._image_path)
        masks, scores, _ = self.segment()
        mask_image = self._get_mask_image(masks)
        self._mat_show.show(mask_image, f"{file_name} optimize")

