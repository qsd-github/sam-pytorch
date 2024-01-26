from abc import ABC


class SAMBase(ABC):
    def image(self, type="matplotlib", is_save=False, save_path=None):
        pass

    def segment(self):
        pass

    def predict(self, type="matplotlib", is_save=False, save_path=None):
        pass

    def save(self, save_path):
        pass