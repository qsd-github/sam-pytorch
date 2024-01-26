import os

import clip
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator

from sam.prompt import SAMPrompt
import cv2 as cv


class SAMText(SAMPrompt):
    def __init__(self, image_path, text, clip_model_path=None, model_type="vit_b", sam_checkpoint_path=None,
                 is_multimask=True, is_cuda=True):
        super().__init__(image_path, text=text, model_type=model_type, sam_checkpoint_path=sam_checkpoint_path, is_multimask=is_multimask, is_cuda=is_cuda)

        self._clip_model_path = clip_model_path

    def image(self, type="matplotlib", is_save=False, save_path=None):
        file_name = os.path.basename(self._image_path)
        if type == "matplotlib":
            self._mat_show.show(self._image, f"{file_name} Text:{self._text}")
        elif type == "opencv":
            image = cv.cvtColor(self._image, cv.COLOR_RGB2BGR)
            self._cv_show.show(image, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")

    def segment(self):
        mask_generator = SamAutomaticMaskGenerator(self._sam_model)
        masks = mask_generator.generate(self._image)
        cropped_boxes = []
        for mask in masks:
            x1, y1, x2, y2 = self.__convert_box_xywh_to_xyxy(mask["bbox"])
            cropped_boxes.append(self.__segment_image(self._image, mask["segmentation"]).astype(np.uint8)[y1:y2, x1:x2])
        if self._is_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                model, preprocess = clip.load(self._clip_model_path, device=device)
            else:
                raise RuntimeError("No GPU found on this computer!")
        else:
            device = torch.device("cpu")
            model, preprocess = clip.load(self._clip_model_path, device=device)

        scores = self.__retriev(model, preprocess, device, cropped_boxes, self._text)
        indices = self.__get_indices_of_values_above_threshold(scores, 0.05)
        return indices, masks

    def predict(self, type="matplotlib", is_save=False, save_path=None):
        file_name = os.path.basename(self._image_path)
        indices, masks = self.segment()
        segmentation_masks = []

        for seg_idx in indices:
            segmentation_mask_image = masks[seg_idx]["segmentation"].astype('uint8') * 255
            segmentation_masks.append(segmentation_mask_image)

        # seg_image = self.image.copy()
        seg_image = np.zeros((self._image.shape[0], self._image.shape[1], 3))
        for segmentation_mask_image in segmentation_masks:
            seg_image[segmentation_mask_image > 0] = [255, 0, 0]
        seg_image_mat = seg_image / 255.
        if type == "matplotlib":
            self._mat_show.show(seg_image_mat, f"{file_name} Text:{self._text}")
        elif type == "opencv":
            image = cv.cvtColor(seg_image.astype("uint8"), cv.COLOR_RGB2BGR)
            self._cv_show.show(image, file_name)
        else:
            raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
        if is_save:
            if save_path is None:
                raise ValueError("Please provide parameters: save_path!")
            image = cv.cvtColor(seg_image.astype("uint8"), cv.COLOR_RGB2BGR)
            cv.imwrite(save_path, image)

    def save(self, save_path):
        indices, masks = self.segment()
        segmentation_masks = []

        for seg_idx in indices:
            segmentation_mask_image = masks[seg_idx]["segmentation"].astype('uint8') * 255
            segmentation_masks.append(segmentation_mask_image)

        # seg_image = self.image.copy()
        seg_image = np.zeros((self._image.shape[0], self._image.shape[1], 3))
        for segmentation_mask_image in segmentation_masks:
            seg_image[segmentation_mask_image > 0] = [255, 0, 0]
        # seg_image_mat = seg_image / 255.
        image = cv.cvtColor(seg_image.astype("uint8"), cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)

    def _prompt_image(self):
        pass

    def _get_mask_image(self, masks):
        pass

    def __getitem__(self, item):
        if item > 0:
            raise IndexError("Text can only return one mask!")
        self.predict()

    def __convert_box_xywh_to_xyxy(self, box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def __segment_image(self, image, segmentation_mask):
        seg_mask = np.array([segmentation_mask, segmentation_mask, segmentation_mask]).transpose(1, 2, 0)
        return np.multiply(image, seg_mask)

    @torch.no_grad()
    def __retriev(self, model, preprocess, device, elements, search_text):
        preprocessed_images = [preprocess(image.astype(dtype=np.uint8)).to(device) for image in elements]
        tokenized_text = clip.tokenize([search_text]).to(device)
        stacked_images = torch.stack(preprocessed_images)
        image_features = model.encode_image(stacked_images)
        text_features = model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = 100. * image_features @ text_features.T
        return probs[:, 0].softmax(dim=0)

    def __get_indices_of_values_above_threshold(self, values, threshold):
        return [i for i, v in enumerate(values) if v > threshold]

