# sam库安装和使用教程

## 简介部分

- `sam`库是根据`meta`公司的`segment-anything`进行了封装，并实现了`text2mask`部分，有任何问题请联系`2472645980@qq.com`。

## 安装部分

- `github`网址：

```python
https://github.com/qsd-github/sam-pytorch
```

- `gitee`网址：

```python
https://gitee.com/qsdmykj/sam-pytorch
```

- `python`版本要求：`python>=3.10.9`

- 先安装`pytorch`系列

```python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117 --default-timeout=10000
```

### 使用.whl安装

```python
pip install sam_pytorch-1.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 使用git源码安装

- 下载源码，然后使用

```python
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 问题：安装`pytorch`系列的时候会报错

- 请直接到`pytorch`官网直接下载`pytorch`和`torchvision`，建议版本`torch==2.0.1+cu117`或更高版本，`torchvision==0.15.2+cu117`或更高版本，安装代码：

```python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117 --default-timeout=10000
```

## 使用部分

### any模式

- 用法：此模式用于将图像中的所有对象进行分割。

#### 方法示例

- 初始化方法(`__init__()`)：

```python
def __init__(self, image_path, model_type="vit_b", sam_checkpoint_path=checkpoint_path, is_cuda=True):
    """
    :param image_path:需要进行分割图片的地址
    :param model_type:sam模型的类别
    :param sam_checkpoint_path:sam权重路径,不提供会自动进行下载
    :param is_cuda:是否使用GPU加速
    """
```

- 原始图片显示(`image()`)

```python
def image(self, type="matplotlib"):
    """
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    """
```

- 图片分割(`segment()`)

```python
def segment(self):
    """
    :return 返回所有的分割信息，包括分割图,面积,分割框,预测iou,预测点,稳定性评分,剪裁框
    """
```

- 图片预测(`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    预测分割图像
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存预测图像
    :param save_path:保存预测图像的位置，如果为真，则填写此参数
    """
```

- 预测图片保存(`save()`)

```python
def save(self, save_path):
    """
    保存预测图片
    :param save_path:保存图像的位置
    """
```

- 所有分割二值图

```python
def segmentations(self):
    """
    :return:分割二值图列表
    """
```

- 所有分割图的面积

```python
def areas(self):
    """
    :return:掩码的面积列表
    """
```

- 所有分割框

```python
def boxes(self):
    """
    :return:分割对象框列表
    """
```

- 所有的预测`IOU`

```python
def predicted_ious(self):
    """
    :return:分割预测IOU列表
    """
```

- 所有的分割点

```python
def point_coords(self):
    """
    :return:分割预测列表
    """
```

- 所有的分割稳定值

```python
def stability_scores(self):
    """
    :return:分割稳定值列表
    """
```

- 所有剪裁框

```python
def crop_boxes(self):
    """
    :return:剪裁框列表, 返回格式为:[X, Y, W, H]
    """
```

#### 完整使用代码

```python
from sam import SAM

sam_any = SAM("./images/truck.jpg",sam_checkpoint_path="./weights/sam_vit_b.pth").any()
sam_any.image()
sam_any.predict(is_save=True, save_path="./outputs/truck_any_mask.jpg")
sam_any.save(save_path="./outputs/truck_any_mask.jpg")
print(sam_any.areas())
```

### points模式

- 用法：此模式是根据点进行分割，前景点标注为`1`，背景点标注为`0`

#### 方法示例

- 初始化方法(`__init__()`)：

```python
def __init__(self, image_path, input_points, input_labels, model_type="vit_b",
                 sam_checkpoint_path="./weights/sam_vit_b.pth", is_multimask=True, is_cuda=True):
    """
    :param image_path:需要进行分割图片的地址
    :param input_points:输入点列表
    :param input_labels:输入标签列表
    :param model_type:sam模型的类别
    :param sam_checkpoint_path:sam权重路径,不提供会自动进行下载
    :param is_multimask:是否是多重mask
    :param is_cuda:是否使用GPU加速
    """
```

- 提示词图片显示(`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存,可设置为True或False
    :param save_path:设置保存地址
    """
```

- 图片分割(`segment()`)

```python
def segment(self):
    """
    :return 返回分割掩码
    """
```

- 图片预测(`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    预测分割图像
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存预测图像
    :param save_path:保存预测图像的位置，如果为真，则填写此参数
    """
```

- 预测图片保存(`save()`)

```python
def save(self, save_path):
    """
    保存预测图片
    :param save_path:保存图像的位置
    """
```

#### 完整使用代码

```python
from sam import SAM
import numpy as np

input_points = np.array([(500, 375), (1125, 625), (500, 70), (1100, 150)])  # 设置了前景点和背景点
input_labels = np.array([1, 1, 0, 0])

sam_points = SAM(r"./images/truck.jpg",sam_checkpoint_path=r"./weights/sam_vit_b.pth", input_points=input_points, input_labels=input_labels).points()
sam_points.image(is_save=True, save_path=r"./outputs/truck_points.jpg")
sam_points.predict(is_save=True, save_path=r"./outputs/truck_points_mask.jpg")
sam_points.save(save_path=r"./outputs/truck_points_mask.jpg")
sam_points[0]  # 若is_multimask设置为True,则有多重掩码,此时可以使用下标进行遍历
```

### box模式

- 用法：此模式是根据框进行分割，在图像中画框之后即可分割框内对象。

#### 方法示例

- 初始化方法(`__init__()`)：

```python
def __init__(self, image_path, input_boxes, model_type="vit_b",
                 sam_checkpoint_path="./weights/sam_vit_b.pth", is_multimask=True, is_cuda=True):
    """
    :param image_path:需要进行分割图片的地址
    :param input_boxes:输入框
    :param model_type:sam模型的类别
    :param sam_checkpoint_path:sam权重路径,不提供会自动进行下载
    :param is_multimask:是否是多重mask
    :param is_cuda:是否使用GPU加速
    """
```

- 提示词图片显示(`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存,可设置为True或False
    :param save_path:设置保存地址
    """
```

- 图片分割(`segment()`)

```python
def segment(self):
    """
    :return 返回分割掩码
    """
```

- 图片预测(`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    预测分割图像
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存预测图像
    :param save_path:保存预测图像的位置，如果为真，则填写此参数
    """
```

- 预测图片保存(`save()`)

```python
def save(self, save_path):
    """
    保存预测图片
    :param save_path:保存图像的位置
    """
```

#### 完整使用代码

```python
from sam import SAM
import numpy as np

input_boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])

sam_box = SAM("./images/truck.jpg",sam_checkpoint_path="./weights/sam_vit_b.pth", input_boxes=input_boxes).boxes()
sam_box.image(is_save=True, save_path="./outputs/truck_box.jpg")
sam_box.predict(is_save=True, save_path="./outputs/truck_box_mask.jpg")
sam_box.save(save_path=r"./outputs/truck_box_mask.jpg")
```

### optim模式

- 用法：此模式用于其他网络优化分割效果不好的掩码

#### 方法示例

- 初始化方法：

```python
def __init__(self, image, optim_masks, model_type="vit_b", sam_checkpoint_path=None, is_multimask=True, is_cuda=True):
    """
    :param optim_masks:待优化的掩码
    :param model_type:sam模型的类别
    :param sam_checkpoint_path:sam权重路径,不提供会自动进行下载
    :param is_multimask:是否是多重mask
    :param is_cuda:是否使用GPU加速
    """
```
- 提示词图片显示(`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存,可设置为True或False
    :param save_path:设置保存地址
    """
```

- 图片分割(`segment()`)

```python
def segment(self):
    """
    :return 返回分割掩码
    """
```

- 图片预测(`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    预测分割图像
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存预测图像
    :param save_path:保存预测图像的位置，如果为真，则填写此参数
    """
```

- 预测图片保存(`save()`)

```python
def save(self, save_path):
    """
    保存预测图片
    :param save_path:保存图像的位置
    """
```

#### 完整使用代码

```python
from sam import SAM
import cv2 as cv

optim_mask = cv.imread("./images/tiger_mask.jpg")
sam_optim = SAM("./images/tiger.jpg", sam_checkpoint_path="./weights/sam_vit_b.pth", optim_masks=optim_mask).optim()
sam_optim.image()
sam_optim.predict(is_save=True, save_path="./outputs/truck_mask_optim.jpg")
sam_optim.save(save_path="./outputs/truck_mask_optim.jpg")
sam_optim[0]
```


### text模式

- 用法：此模式是根据文本进行分割。

- 使用此方法前先对`openai-clip`中的`clip.py`文件进行修改，修改位置如下：

  - 第`10`行，添加`ToPILImage`

    ```python
    from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
    ```

  - 第`81`行，添加`ToPILImage(),`

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

  - 使`openai-clip`能够支持`opencv`。


#### 方法示例

- 初始化方法(`__init__()`)：

```python
def __init__(self, image_path, text, clip_model_path=None, model_type="vit_b", sam_checkpoint_path=None,
                 is_multimask=True, is_cuda=True):
    """
    :param image_path:需要进行分割图片的地址
    :param text:输入的文本
    :param clip_model_path:clip的权重，可以选择
    	RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    	一般是使用'ViT-B/32'，若有权重可以使用权重地址
    :param model_type:sam模型的类别
    :param sam_checkpoint_path:sam权重路径,不提供会自动进行下载
    :param is_multimask:是否是多重mask
    :param is_cuda:是否使用GPU加速
    """
```

- 提示词图片显示(`image()`)

```python
def image(self, type="matplotlib", is_save=False, save_path=None):
    """
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存,可设置为True或False
    :param save_path:设置保存地址
    """
```

- 图片分割(`segment()`)

```python
def segment(self):
    """
    :return 返回分割掩码
    """
```

- 图片预测(`predict()`)

```python
def predict(self, type="matplotlib", is_save=False, save_path=None):
    """
    预测分割图像
    :param type:可以在"matplotlib"和"opencv"两个选项中进行选择
    :param is_save:是否保存预测图像
    :param save_path:保存预测图像的位置，如果为真，则填写此参数
    """
```

- 预测图片保存(`save()`)

```python
def save(self, save_path):
    """
    保存预测图片
    :param save_path:保存图像的位置
    """
```

#### 完整使用代码

```python
from sam import SAM

sam_text = SAM("./images/fruits.jpg", sam_checkpoint_path="./weights/sam_vit_b.pth", text="orange", clip_model_path='ViT-B/32').text()
sam_text.image() 
sam_text.predict(is_save=True, save_path=r"./outputs/fruits_text_mask.jpg")
sam_text.save(save_path=r"./outputs/fruits_text_mask.jpg")
```

