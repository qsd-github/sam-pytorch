from abc import ABC


class Display(ABC):

    def show(self, image, title=""):
        pass

    def connect_show(self, image, mask_image, image_title="", mask_title=""):
        pass