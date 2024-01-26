import os
from segment_anything import SamPredictor
import cv2 as cv
from sam.prompt import SAMPrompt

project_path = os.path.dirname(__file__)


class SAMPoints(SAMPrompt):
    def __init__(self, image_path, input_points, input_labels, model_type="vit_b",
                 sam_checkpoint_path=None, is_multimask=True, is_cuda=True):
        super().__init__(image_path, input_points=input_points, input_labels=input_labels, model_type=model_type, sam_checkpoint_path=sam_checkpoint_path, is_multimask=is_multimask, is_cuda=is_cuda)

    # self.__image_path = image_path
        # self.__image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        #
        # self.__input_points = input_points
        # self.__input_labels = input_labels
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
    #     file_name = os.path.basename(self._image_path)
    #     points_image = self._prompt_image()
    #     if type == "matplotlib":
    #         self._mat_show.show(points_image, file_name)
    #     elif type == "opencv":
    #         image = cv.cvtColor(points_image, cv.COLOR_RGB2BGR)
    #         self._cv_show.show(image, file_name)
    #     else:
    #         raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
    #     if is_save:
    #         if save_path is None:
    #             raise ValueError("Please provide parameters: save_path!")
    #         image = cv.cvtColor(points_image, cv.COLOR_RGB2BGR)
    #         cv.imwrite(save_path, image)

    def segment(self):
        """
        Predicting through points and returning masks and evaluation metrics
        :return:
            masks: Mask predicted by points
            scores: Mask prediction evaluation indicators
            logits: logits value
        """
        predictor = SamPredictor(self._sam_model)
        predictor.set_image(self._image)
        masks, scores, logits = predictor.predict(
            point_coords=self._input_points,
            point_labels=self._input_labels,
            multimask_output=self._is_multimask,
        )
        # predictor = SamPredictor(self.__sam_model)
        # predictor.set_image(self.__image)
        # masks, scores, logits = predictor.predict(
        #     point_coords=self.__input_points,
        #     point_labels=self.__input_labels,
        #     multimask_output=self.__is_multimask,
        # )
        return masks, scores, logits

    # def predict(self, type="matplotlib", is_save=False, save_path=None):
    #     file_name = os.path.basename(self.__image_path)
    #     masks, scores, _ = self.segment()
    #     for i, (mask, score) in enumerate(zip(masks, scores)):
    #         mask_image = self._get_mask_image(mask)
    #         if type == "matplotlib":
    #             self.__mat_show.show(mask_image, file_name)
    #         elif type == "opencv":
    #             self.__cv_show.show(mask_image, file_name)
    #         else:
    #             raise ValueError("The value of the type parameter should be selected in ['matplotlib','opencv']")
    #         if is_save:
    #             if save_path is None:
    #                 raise ValueError("When is_save selects True, output_path cannot be None!")
    #             save_path_list = save_path.split(".")
    #             save_path_f = save_path_list[0] + str(i) + ".jpg"
    #             cv.imwrite(save_path_f, mask_image * 255)

    # def save(self, save_path):
    #     masks, scores, _ = self.segment()
    #     for i, (mask, score) in enumerate(zip(masks, scores)):
    #         mask_image = self._get_mask_image(mask)
    #         save_path_list = save_path.split(".")
    #         save_path_f = save_path_list[0] + str(i) + ".jpg"
    #         cv.imwrite(save_path_f, mask_image * 255)

    def _prompt_image(self):
        points_info = list(zip(self._input_points, self._input_labels))
        img = self._image.copy()
        radius = 10
        thickness = -1
        # 画圆
        for point_info in points_info:
            center_point, label = point_info
            # 指定颜色和线宽
            if label == 1:
                color = (255, 0, 0)  # RGB格式，这里是红色
            elif label == 0:
                color = (0, 0, 0)  # RGB格式，这里是黑色
            cv.circle(img, center_point, radius, color, thickness)
        return img

    # def __getitem__(self, item):
    #     file_name = os.path.basename(self.__image_path)
    #     masks, scores, _ = self.segment()
    #     if item > len(masks):
    #         raise IndexError("The index value should be less than the number of masks!")
    #     mask_image = self._get_mask_image(masks[item])
    #     self.__mat_show.show(mask_image, file_name)