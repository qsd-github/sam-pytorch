from sam.utils.display.base import Display
import matplotlib.pyplot as plt
import cv2 as cv


class Matplotlib(Display):
    def __init__(self, figsize=(10, 10), fontsize=20, is_axis=False):
        self.figsize = figsize
        self.fontsize = fontsize
        self.is_axis = is_axis

    def show(self, image, title=""):
        plt.figure(figsize=self.figsize)
        plt.title(title, fontsize=self.fontsize)
        plt.imshow(image)
        if self.is_axis:
            plt.axis('on')
        else:
            plt.axis('off')
        plt.show()

    def connect_show(self, image, mask_image, image_title="", mask_title=""):
        plt.subplot(1, 2, 1)
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(image_title)

        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(mask_image, cv.COLOR_BGR2RGB))
        plt.title(mask_title)

        # 调整子图之间的间距
        plt.tight_layout()

        # 显示图像
        plt.show()


