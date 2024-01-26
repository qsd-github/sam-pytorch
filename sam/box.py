import os

import cv2 as cv

import torch
from segment_anything import SamPredictor

from sam.prompt import SAMPrompt

project_path = os.path.dirname(__file__)


class SAMBox(SAMPrompt):
    def __init__(self, image_path, input_boxes, model_type="vit_b", sam_checkpoint_path=None,
                 is_multimask=True, is_cuda=True):

        super().__init__(image_path, input_boxes=input_boxes, model_type=model_type, sam_checkpoint_path=sam_checkpoint_path, is_multimask=is_multimask, is_cuda=is_cuda)

        # self.__image_path = image_path
        # self.__image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        #
        # self.__input_boxes = input_boxes
        # self.__is_multimask = is_multimask
        #
        # model_url = {
        #     "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        #     "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        #     "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        # }
        # self.__model_type = model_type
        # self.__sam_checkpoint_path = sam_checkpoint_path
        # self.__is_cuda = is_cuda
        #
        # weight = os.listdir(os.path.join(project_path, "weights"))
        # if "sam_" + model_type + ".pth" not in weight:
        #     Weight.download(model_type, model_url[model_type],
        #                     os.path.join(project_path, "weights", "sam_" + model_type + ".pth"))
        #
        # if self.__is_cuda:
        #     if torch.cuda.is_available():
        #         device = torch.device("cuda:0")
        #         self.__sam_model = sam_model_registry[model_type](checkpoint=self.__sam_checkpoint_path).to(
        #             device=device)
        #     else:
        #         raise RuntimeError("No GPU found on this computer!")
        # else:
        #     self.__sam_model = sam_model_registry[model_type](checkpoint=self.__sam_checkpoint_path)
        #
        # self.__mat_show = Matplotlib()
        # self.__cv_show = OpenCV()

    # def image(self, type="matplotlib", is_save=None, save_path=None):
    #     file_name = os.path.basename(self.__image_path)
    #     boxes_image = self._prompt_image()
    #     if type == "matplotlib":
    #         self.__mat_show.show(boxes_image, file_name)
    #     elif type == "opencv":
    #         image = cv.cvtColor(boxes_image, cv.COLOR_RGB2BGR)
    #         self.__cv_show.show(image, file_name)
    #     else:
    #         raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
    #     if is_save:
    #         if save_path is None:
    #             raise ValueError("Please provide parameters: save_path!")
    #         image = cv.cvtColor(boxes_image, cv.COLOR_RGB2BGR)
    #         cv.imwrite(save_path, image)

    def segment(self):
        """
        Segment anything by boxes
        :return:
            masks: Mask predicted by points
            scores: Mask prediction evaluation indicators
            logits: logits value
        """
        if self._is_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                boxes = torch.tensor(self._input_boxes, device=device)
            else:
                raise RuntimeError("No GPU found on this computer!")
        else:
            device = torch.device("cpu")
            boxes = torch.tensor(self._input_boxes, device=device)

        predictor = SamPredictor(self._sam_model)
        predictor.set_image(self._image)

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, self._image.shape[:2])
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self._is_multimask,
        )
        return masks[:, 0:1, :, :].cpu().numpy(), scores[0], logits

    # def predict(self, type="matplotlib", is_save=False, save_path=None):
    #     file_name = os.path.basename(self._image_path)
    #     masks, scores, _ = self.segment()
    #     for i, (mask, score) in enumerate(zip(masks, scores)):
    #         mask_image = self._get_mask_image(mask[0].cpu().numpy())
    #         if type == "matplotlib":
    #             self._mat_show.show(mask_image, f"{file_name} Mask {i + 1}, Score: {score[0].item():.3f}")
    #         elif type == "opencv":
    #             self._cv_show.show(mask_image, file_name)
    #         else:
    #             raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
    #         if is_save:
    #             if save_path is None:
    #                 raise ValueError("When is_save selects True, save_path cannot be None!")
    #             save_path_list = save_path.split(".")
    #             save_path_f = save_path_list[0] + str(i) + ".jpg"
    #             cv.imwrite(save_path_f, mask_image * 255)
    #
    # def save(self, save_path):
    #     masks, scores, _ = self.segment()
    #     for i, (mask, score) in enumerate(zip(masks, scores)):
    #         mask_image = self._get_mask_image(mask[0])
    #         save_path_list = save_path.split(".")
    #         save_path_f = save_path_list[0] + str(i) + ".jpg"
    #         cv.imwrite(save_path_f, mask_image * 255)

    def _prompt_image(self):
        img = self._image.copy()
        color = (255, 0, 0)  # RGB格式，这里是红色
        thickness = 2
        for box in self._input_boxes:
            start_point, end_point = (box[0], box[1]), (box[2], box[3])
            cv.rectangle(img, start_point, end_point, color, thickness)
        return img





