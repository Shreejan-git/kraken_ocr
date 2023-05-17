import random

import numpy as np
from kraken import blla
from kraken.lib import vgsl
from PIL import Image, ImageOps
import cv2


def visualize_polylines(cv_image, lines):
    """

    :param cv_image: image read with cv2 to draw the polylines. PIL read image is not accepted
    :param baseline_seg:
    :return:
    """
    # for lines in baseline_seg['lines']:
    for baseline in lines:
        base_line = baseline['baseline']
        base_line = np.array(base_line, np.int32)
        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        cv_image = cv2.polylines(cv_image, [base_line],
                                 False, (r, g, b), 10)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = '/home/dell/Documents/handwritten_images/testingimages/s1.jpg'
    # model_path = 'path/to/model/file'
    # model = vgsl.TorchVGSLModel.load_model(model_path)
    # image = Image.open(image_path)
    # image = ImageOps.exif_transpose(image)
    cv_image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv_image)  # reading with pil as a requirement of Kraken.
    # pil_image.show()
    baseline_seg = blla.segment(pil_image, model=None, device='cpu')  # Baseline segmenter
    visualize_polylines(cv_image, baseline_seg['lines'])

