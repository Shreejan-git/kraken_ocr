import random

import numpy as np
from kraken import blla
from kraken.lib import vgsl
from PIL import Image, ImageOps
import cv2


def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BLURRING (Doing two layered blurring)
    diameter = 50
    sigma_color = 55
    sigma_space = 50
    bilateral_blurred = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)

    # binarization_low = cv2.adaptiveThreshold(bilateral_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                          cv2.THRESH_BINARY_INV,
    #                                          21, 11)

    # median_blurred_high = cv2.medianBlur(binarization_high, 3)
    # median_blurred_low = cv2.medianBlur(binarization_low, 3)

    # # Erosion
    # eroded_img_high = cv2.erode(binarization_high, None)
    # eroded_img_low = cv2.erode(binarization_low, None)

    # dilated_img_high = cv2.dilate(median_blurred_high, None)
    dilated_img_low = cv2.dilate(bilateral_blurred, None, iterations=2)

    # cv2.namedWindow('low', cv2.WINDOW_NORMAL)
    # cv2.imshow('low', bilateral_blurred)
    #
    # cv2.waitKey(0)

    return dilated_img_low


def visualize_polylines(img, cv_image, lines):
    """

    :param cv_image: image read with cv2 to draw the polylines. PIL read image is not accepted
    :param baseline_seg:
    :return:
    """
    # for lines in baseline_seg['lines']:
    if lines:
        for baseline in lines:
            base_line = baseline['baseline']
            base_line = np.array(base_line, np.int32)
            r = random.randint(0, 256)
            g = random.randint(0, 256)
            b = random.randint(0, 256)
            cv_image = cv2.polylines(cv_image, [base_line],
                                     False, (r, g, b), 10)

        cv2.namedWindow(f'{img}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'{img}', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('Could not detect anything')


def draw_polygon(cv_image, lines):
    """
    Draws polygons above the detected words
    :param cv_image:
    :param lines:
    :return:
    """
    if lines:
        for baseline in lines:
            base_line = baseline['boundary']
            base_line = np.array(base_line, np.int32)
            # r = random.randint(0, 256)
            # g = random.randint(0, 256)
            # b = random.randint(0, 256)
            cv2.fillPoly(cv_image, pts=[base_line], color=(228, 242, 210))

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('could not detect anything.')


def image_preprocessing(img):
    """
    Image is already in gray format.
    """
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BLURRING (Doing two layered blurring)
    diameter = 50
    sigma_color = 55
    sigma_space = 50
    bilateral_blurred = cv2.bilateralFilter(gray_scale, diameter, sigma_color, sigma_space)
    # median_blurred = cv2.medianBlur(bilateral_blurred, 3)
    #
    # # BINARIZATION
    # binarization_high = cv2.adaptiveThreshold(bilateral_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                           cv2.THRESH_BINARY_INV,
    #                                           23, 11)

    binarization_low = cv2.adaptiveThreshold(bilateral_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV,
                                             21, 11)

    # median_blurred_high = cv2.medianBlur(binarization_high, 3)
    median_blurred_low = cv2.medianBlur(binarization_low, 3)

    # # Erosion
    # eroded_img_high = cv2.erode(binarization_high, None)
    # eroded_img_low = cv2.erode(binarization_low, None)

    # dilated_img_high = cv2.dilate(median_blurred_high, None)
    dilated_img_low = cv2.dilate(median_blurred_low, None, iterations=2)

    # normalized_img = eroded_img/255.0

    #
    # cv2.namedWindow('high', cv2.WINDOW_NORMAL)
    # cv2.imshow('high', dilated_img_high)

    cv2.namedWindow('low', cv2.WINDOW_NORMAL)
    cv2.imshow('low', dilated_img_low)

    cv2.waitKey(0)

    return dilated_img_low


if __name__ == "__main__":
    import os

    dir_path = '/home/dell/Documents/handwritten_images/testingimages'

    for img in os.listdir(dir_path):
        image_path = os.path.join(dir_path, img)
        print(image_path)

        # image_path = '/home/dell/Documents/handwritten_images/testingimages/s1.jpg'
        # model_path = 'path/to/model/file'
        # model = vgsl.TorchVGSLModel.load_model(model_path)
        # image = Image.open(image_path)
        # image = ImageOps.exif_transpose(image)
        cv_image = cv2.imread(image_path)
        # preprocessed = preprocessing(cv_image)
        pil_image = Image.fromarray(cv_image)  # reading with pil as a requirement of Kraken.
        # pil_image.show()
        baseline_seg = blla.segment(pil_image, model=None, device='cpu')  # Baseline segmenter
        visualize_polylines(img, cv_image, baseline_seg['lines'])
        draw_polygon(cv_image, baseline_seg['lines'])

