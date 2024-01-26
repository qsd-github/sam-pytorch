# Tutorial for Installing and Using the sam Library

## Introduction

- The `sam` library is a wrapper based on the `segment-anything` module from the `meta` company and implements the `text2mask` part. For any inquiries, please contact `2472645980@qq.com`.

## Installation

- `Python` version requirement: `python>=3.10.9`

### Installation using .whl file

```python
pip install sam_pytorch-1.0-py3-none-any.whl
```

### Installation using Git source code

```python
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Issue: Error occurs during PyTorch installation.

- Please download `PyTorch` and `torchvision` directly from the `PyTorch` official website. It is recommended to use versions `torch==2.0.1+cu117` or higher and `torchvision==0.15.2+cu117` or higher. Install with the following code:

```python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

## Usage

### any mode

- Usage: This mode is used to segment all objects in the image.

#### Method Examples

- Initialization method (`__init__()`):

```python
def __init__(self, image_path, model_type="vit_b", sam_checkpoint_path=checkpoint_path, is_cuda=True):
    """
    :param image_path: Path to the image to be segmented
    :param model_type: Type of sam model
    :param sam_checkpoint_path: Path to the sam weights, automatically downloaded if not provided
    :param is_cuda: Whether to use GPU acceleration
    """
```

- Display original image (`image()`)

```python
def image(self, type="matplotlib"):
    """
    :param type: Choose between "matplotlib" and "opencv"
    """
```

- Image segmentation (`segment()`)

```python
def segment(self):
    """
    :return: Returns all segmentation information, including segmented image, area, bounding box, predicted IOU, predicted points, stability score, cropping box
    """
```

- Image prediction (`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    Predict segmented image
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save the predicted image
    :param save_path: Save path for the predicted image, fill in this parameter if is_save is True
    """
```

- Save predicted image (`save()`)

```python
def save(self, save_path):
    """
    Save the predicted image
    :param save_path: Save location for the image
    """
```

- Binary images for all segments

```python
def segmentations(self):
    """
    :return: List of binary segmented images
    """
```

- Areas of all segmented images

```python
def areas(self):
    """
    :return: List of mask areas
    """
```

- All segmented boxes

```python
def boxes(self):
    """
    :return: List of segmented object boxes
    """
```

- All predicted `IOU` values

```python
def predicted_ious(self):
    """
    :return: List of segmentation prediction IOU values
    """
```

- All segmentation points

```python
def point_coords(self):
    """
    :return: List of segmentation prediction points
    """
```

- All segmentation stability values

```python
def stability_scores(self):
    """
    :return: List of segmentation stability values
    """
```

- All cropping boxes

```python
def crop_boxes(self):
    """
    :return: List of cropping boxes, format: [X, Y, W, H]
    """
```

#### Complete Usage Code

```python
from sam.sam import SAM

sam_any = SAM("./images/truck.jpg",sam_checkpoint_path="./weights/sam_vit_b.pth").any()
sam_any.image()
sam_any.predict(is_save=True, save_path="./outputs/truck_any_mask.jpg")
sam_any.save(save_path="./outputs/truck_any_mask.jpg")
print(sam_any.areas())
```

### points mode

- Usage: This mode segments based on points, with foreground points labeled as `1` and background points labeled as `0`.

#### Method Examples

- Initialization method (`__init__()`):

```python
def __init__(self, image_path, input_points, input_labels, model_type="vit_b",
                 sam_checkpoint_path="./weights/sam_vit_b.pth", is_multimask=True, is_cuda=True):
    """
    :param image_path: Path to the image to be segmented
    :param input_points: List of input points
    :param input_labels: List of input labels
    :param model_type: Type of sam model
    :param sam_checkpoint_path: Path to the sam weights, automatically downloaded if not provided
    :param is_multimask: Whether it is multimask
    :param is_cuda: Whether to use GPU acceleration
    """
```

- Prompt Word Image Display (`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save, set to True or False
    :param save_path: Set the save address
    """
```

- Image segmentation (`segment()`)

```python
def segment(self):
    """
    :return: Returns the segmentation mask
    """
```

- Image prediction (`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    Predict segmented image
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save the predicted image
    :param save_path: Save path for the predicted image, fill in this parameter if is_save is True
    """
```

- Save predicted image (`save()`)

```python
def save(self, save_path):
    """
    Save the predicted image
    :param save_path: Save location for the image
    """
```

#### Complete Usage Code

```python
from sam.sam import SAM
import numpy as np

input_points = np.array([(500, 375), (1125, 625), (500, 70), (1100, 150)])  # Set foreground and background points
input_labels = np.array([1, 1, 0, 0])

sam_points = SAM(r"./images/truck.jpg",sam_checkpoint_path=r"./weights/sam_vit_b.pth", input_points=input_points, input_labels=input_labels).points()
sam_points.image(is_save=True, save_path=r"./outputs/truck_points.jpg")
sam_points.predict(is_save=True, save_path=r"./outputs/truck_points_mask.jpg")
sam_points.save(save_path=r"./outputs/truck_points_mask.jpg")
sam_points[0]  # If is_multimask is set to True, there are multiple masks, you can iterate using an index
```

### box mode

- Usage: This mode segments based on boxes, and objects inside the drawn boxes can be segmented.

#### Method Examples

- Initialization method (`__init__()`)：

```python
def __init__(self, image_path, input_boxes, model_type="vit_b",
                 sam_checkpoint_path="./weights/sam_vit_b.pth", is_multimask=True, is_cuda=True):
    """
    :param image_path: Path to the image to be segmented
    :param input_boxes: Input boxes
    :param model_type: Type of sam model
    :param sam_checkpoint_path: Path to the sam weights, automatically downloaded if not provided
    :param is_multimask: Whether it is multimask
    :param is_cuda: Whether to use GPU acceleration
    """
```

- Prompt Word Image Display (`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save, set to True or False
    :param save_path: Set the save address
    """
```

- Image segmentation (`segment()`)

```python
def segment(self):
    """
    :return: Returns the segmentation mask
    """
```

- Image prediction (`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    Predict segmented image
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save the predicted image
    :param save_path: Save path for the predicted image, fill in this parameter if is_save is True
    """
```

- Save predicted image (`save()`)

```python
def save(self, save_path):
    """
    Save the predicted image
    :param save_path: Save location for the image
    """
```

#### Complete Usage Code

```python
from sam.sam import SAM
import numpy as np

input_boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])

sam_box = SAM("./images/truck.jpg",sam_checkpoint_path="./weights/sam_vit_b.pth", input_boxes=input_boxes).box()
sam_box.image(is_save=True, save_path="./outputs/truck_box.jpg")
sam_box.predict(is_save=True, save_path="./outputs/truck_box_mask.jpg")
sam_box.save(save_path=r"./outputs/truck_box_mask.jpg")
```

### optim mode

- Usage: This mode is used to optimize masks that do not perform well with other network segmentation results.

#### Method Examples

- Initialization method:

```python
def __init__(self, image, optim_masks, model_type="vit_b", sam_checkpoint_path=None, is_multimask=True, is_cuda=True):
    """
    :param optim_masks: Masks to be optimized
    :param model_type: Type of sam model
    :param sam_checkpoint_path: Path to the sam weights, automatically downloaded if not provided
    :param is_multimask: Whether it is multimask
    :param is_cuda: Whether to use GPU acceleration
    """
```

- Display original image (`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save, set to True or False
    :param save_path: Set the save address
    """
```

- Image segmentation (`segment()`)

```python
def segment(self):
    """
    :return: Returns the segmentation mask
    """
```

- Image prediction (`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    Predict segmented image
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save the predicted image
    :param save_path: Save path for the predicted image, fill in this parameter if is_save is True
    """
```

- Save predicted image (`save()`)

```python
def save(self, save_path):
    """
    Save the predicted image
    :param save_path: Save location for the image
    """
```

#### Complete Usage Code

```python
from sam.sam import SAM

optim_mask = cv.imread("./images/tiger_mask.jpg")
sam_optim = SAM("./images/tiger.jpg", sam_checkpoint_path="./weights/sam_vit_b.pth", optim_masks=optim_mask).optim()
sam_optim.image()
sam_optim.predict(is_save=True, save_path="./outputs/truck_mask_optim.jpg")
sam_optim.save(save_path="./outputs/truck_mask_optim.jpg")
sam_optim[0]
```

### text mode

- Usage: This mode is based on text for segmentation.

- Before using this method, modify the `clip.py` file in `openai-clip` at the following locations:

  - Line `10`, add `ToPILImage`

    ```python
    from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
    ```

  - Line `81`, add `ToPILImage(),`

    ```python
    return Compose([
        ToPILImage(),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    ```

  - This modification allows `openai-clip` to support `opencv`.

#### Method Examples

- Initialization method (`__init__()`):

```python
def __init__(self, image_path, text, model_type="vit_b",
                 sam_checkpoint_path="./weights/sam_vit_b.pth", is_multimask=True, is_cuda=True):
    """
    :param image_path: Path to the image to be segmented
    :param text: Input text
    :param model_type: Type of sam model
    :param sam_checkpoint_path: Path to the sam weights, automatically downloaded if not provided
    :param is_multimask: Whether it is multimask
    :param is_cuda: Whether to use GPU acceleration
    """
```

- Display original image (`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type: Choose between "matplotlib" and "opencv"
    :param is_save: Whether to save, set to True or False
    :param save_path: Set the save address
    """
```

- Image segmentation (`segment()`)

```python
def segment(self):
    """
    :return: Returns the segmentation mask
    """
```

- Image prediction (`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    Predicts segmented images.
    :param type: Choose between "matplotlib" and "opencv".
    :param is_save: Whether to save the predicted images.
    :param save_path: Location to save the predicted images, fill this parameter if is_save is True.
    """
```

- Save Predicted Images (`save()`)

~~~python
def save(self, save_path):
    """
    Saves predicted images.
    :param save_path: Location to save the images.
    """
~~~

#### Complete Usage Code

```python
from sam.sam import SAM

sam_text = SAM("./images/fruits.jpg", sam_checkpoint_path="./weights/sam_vit_b.pth", text="orange", clip_model_path=r"./weights/ViT-B-32.pt").text()
sam_text.image() 
sam_text.predict(is_save=True, save_path=r"./outputs/fruits_text_mask.jpg")
sam_text.save(save_path=r"./outputs/fruits_text_mask.jpg")
```
