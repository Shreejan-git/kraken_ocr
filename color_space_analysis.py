import os

import cv2
from PIL import Image



import cv2
import numpy as np

def histogram_equalization_hsv(image):
    """
    applying histogram equalization only on Value (v) of HSV
    :param image: input image
    :return:
    """
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split HSV image into individual components
    h, s, v = cv2.split(hsv_image)

    # Apply histogram equalization to the value component
    # equalized_v = cv2.equalizeHist(v)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    equalized_v = clahe.apply(v)

    # Merge the equalized value component with original hue and saturation components
    equalized_hsv = cv2.merge([h, s, equalized_v])

    # Convert the equalized HSV image back to BGR color space
    equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)

    return equalized_image


if __name__ == "__main__":

    image_path = '/home/dell/Documents/handwritten_images/testingimages/s43.jpg'

    print(image_path)

    # image_pil = Image.open(image_path)
    image = cv2.imread(image_path)
    t = histogram_equalization_hsv(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # H = image_hsv.copy()
    # H[:, :, (1, 2)] = 255  # set S and V channels to max (255)
    # H_RGB = cv2.cvtColor(H, cv2.COLOR_HSV2RGB)  # convert back to RGB
    # # plt.show()
    # # plt.imshow(H_RGB)
    #
    # S = image_hsv.copy()
    # S[:, :, 0] = 179  # set H to max (179)
    # S[:, :, 2] = 255  # set V to max (179)
    # S_RGB = cv2.cvtColor(S, cv2.COLOR_HSV2RGB)  # convert back to RGB

    # V = image_hsv.copy()
    # V[:, :, 0] = 179  # set H to max (179)
    # V[:, :, 1] = 0  # set S to 0
    # V_RGB = cv2.cvtColor(V, cv2.COLOR_HSV2RGB)  # convert back to RGB
    # plt.show()
    # plt.imshow(V_RGB)

    # image_pil.show('PIL IMAGE')

    # cv2.namedWindow('cv2 H', cv2.WINDOW_NORMAL)
    # cv2.imshow('cv2 H', H_RGB)
    #
    # cv2.namedWindow('cv2 s', cv2.WINDOW_NORMAL)
    # cv2.imshow('cv2 s', S_RGB)

    # cv2.namedWindow('cv2 gray', cv2.WINDOW_NORMAL)
    # cv2.imshow('cv2 gray', gray)

    cv2.namedWindow('cv2 V', cv2.WINDOW_NORMAL)
    cv2.imshow('cv2 V', t)

    cv2.waitKey(0)
    # image_pil.close()
