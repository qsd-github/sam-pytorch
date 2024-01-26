from sam.box import SAMBox
from sam.optim import SAMOptim
from sam.points import SAMPoints
from sam.seganything import SAMAny
from sam.text import SAMText


class SAM:
    def __init__(self, image_path, input_points=None,
                 input_labels=None, input_boxes=None, optim_masks=None, text=None, model_type="vit_b", sam_checkpoint_path=None, clip_model_path=None, is_multimask=True, is_cuda=True):
        if image_path is None:
            raise ValueError("Please provide parameters: ['image_path']!")
        self._image_path = image_path
        self._input_points = input_points
        self._input_labels = input_labels
        self._input_boxes = input_boxes
        self._optim_masks = optim_masks
        self._text = text
        self._model_type = model_type
        self._sam_checkpoint_path = sam_checkpoint_path
        self._clip_model_path = clip_model_path
        self._is_multimask = is_multimask
        self._is_cuda = is_cuda

    def any(self):
        return SAMAny(self._image_path, self._model_type, self._sam_checkpoint_path, self._is_cuda)

    def points(self):
        if self._input_points is None or self._input_labels is None:
            raise ValueError("The points segmentation mode requires parameters to be provided:['input_points', 'input_labels']!")
        return SAMPoints(self._image_path, self._input_points, self._input_labels, self._model_type, self._sam_checkpoint_path, self._is_multimask, self._is_cuda)

    def boxes(self):
        if self._input_boxes is None:
            raise ValueError("The boxes segmentation mode requires parameters to be provided:['input_boxes']!")
        return SAMBox(self._image_path, self._input_boxes, self._model_type, self._sam_checkpoint_path, self._is_multimask, self._is_cuda)

    def optim(self):
        if self._optim_masks is None:
            raise ValueError("The optim segmentation mode requires parameters to be provided:['optim_masks']!")
        return SAMOptim(self._image_path, self._optim_masks, self._model_type, self._sam_checkpoint_path, self._is_multimask, self._is_cuda)

    def text(self):
        if self._text is None:
            raise ValueError("The optim segmentation mode requires parameters to be provided:['text']!")
        return SAMText(self._image_path, self._text, self._clip_model_path, self._model_type, self._sam_checkpoint_path,
                 self._is_multimask, self._is_cuda)