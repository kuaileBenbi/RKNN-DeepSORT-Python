import numpy as np
import cv2

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, names=None, identities=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = "{:}{:d}".format(names[i], id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        text_y1 = (
            y1 - t_size[1] - 10 if y1 - t_size[1] - 10 > 0 else y1 + t_size[1] + 10
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, text_y1 - t_size[1] - 4), (x1 + t_size[0] + 3, text_y1), color, -1
        )
        cv2.putText(
            img, label, (x1, text_y1 - 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2
        )
    return img


if __name__ == "__main__":
    for i in range(82):
        print(compute_color_for_labels(i))
