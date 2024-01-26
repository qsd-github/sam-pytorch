import cv2 as cv
from sam.utils.display.base import Display


class OpenCV(Display):
    def show(self, image, title=""):
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def connect_show(self, image, mask_image, image_title="", mask_title=""):
        min_height = min(image.shape[0], mask_image.shape[0])
        image1 = image[:min_height, :].copy()
        image2 = mask_image[:min_height, :].copy()

        # 水平拼接两张图片
        combined_image = cv.cvtColor(cv.hconcat([image1, image2]), cv.COLOR_RGB2BGR)

        # 显示拼接后的图片
        cv.imshow(image_title, combined_image)
        cv.waitKey(0)
        cv.destroyAllWindows()