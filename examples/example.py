import os

import numpy as np
import cv2 as cv

from sam.sam import SAM

images_path = os.path.join(os.path.dirname(__file__), "../images")
weights_path = os.path.join(os.path.dirname(__file__), "../weights")
outputs_path = os.path.join(os.path.dirname(__file__), "../outputs")


if __name__ == '__main__':
    input_points = np.array([(500, 375), (1125, 625), (500, 70), (1100, 150)])  # 设置了前景点和背景点
    input_labels = np.array([1, 1, 0, 0])

    input_boxes = np.array([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ])

    # sam_any = SAM(os.path.join(images_path, "truck.jpg"), sam_checkpoint_path=os.path.join(weights_path, "sam_vit_b.pth")).any()
    # sam_any.image()
    # sam_any.predict(is_save=True, save_path=os.path.join(outputs_path, "truck_any_mask.jpg"))
    # sam_any.save(save_path=os.path.join(outputs_path, "truck_any_mask.jpg"))
    # print(sam_any.areas())

    # sam_points = SAM(os.path.join(images_path, "truck.jpg"), sam_checkpoint_path=os.path.join(weights_path, "sam_vit_b.pth"), input_points=input_points, input_labels=input_labels).points()
    # sam_points.image(is_save=True, save_path=os.path.join(outputs_path, "truck_points.jpg"))
    # sam_points.predict(is_save=True, save_path=os.path.join(outputs_path, "truck_points_mask.jpg"))
    # sam_points.save(save_path=os.path.join(outputs_path, "truck_points_mask.jpg"))
    # sam_points[0]

    # sam_box = SAM(os.path.join(images_path, "truck.jpg"), sam_checkpoint_path=os.path.join(weights_path, "sam_vit_b.pth"),input_boxes=input_boxes).boxes()
    # sam_box.image(is_save=True, save_path=os.path.join(outputs_path, "truck_boxes.jpg"))
    # sam_box.predict(is_save=True, save_path=os.path.join(outputs_path, "truck_boxes_mask.jpg"))
    # sam_box.save(save_path=os.path.join(outputs_path, "truck_boxes_mask.jpg"))
    # sam_box[0]

    optim_mask = cv.imread(os.path.join(images_path, "tiger_mask.jpg"))

    sam_optim = SAM(os.path.join(images_path, "tiger.jpg"), sam_checkpoint_path=os.path.join(weights_path, "sam_vit_b.pth"), optim_masks=optim_mask).optim()
    sam_optim.image()
    # sam_optim.predict(is_save=True, save_path=os.path.join(outputs_path, "truck_mask_optim.jpg"))
    # sam_optim.save(save_path=os.path.join(outputs_path, "truck_mask_optim.jpg"))
    # sam_optim[0]

    # sam_text = SAM(os.path.join(images_path, "fruits.jpg"), sam_checkpoint_path=os.path.join(weights_path, "sam_vit_b.pth"), text="orange", clip_model_path=r"C:\Users\24726\Desktop\Segment Anything\code\weights\ViT-B-32.pt").text()
    # sam_text.image()
    # sam_text.predict(is_save=True, save_path=os.path.join(outputs_path, "fruits_fruits_mask.jpg"))
    # sam_text.save(save_path=os.path.join(outputs_path, "fruits_fruits_mask.jpg"))
    # sam_text[0]