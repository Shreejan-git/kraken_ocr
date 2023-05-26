import random

import numpy as np
from kraken import blla
from PIL import Image, ImageOps
import cv2
# from color_space_analysis import histogram_equalization_hsv
from kraken.binarization import nlbin


def preprocessing(im):

    binarized_image = nlbin(im)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # BLURRING (Doing two layered blurring)
    # diameter = 50
    # sigma_color = 55
    # sigma_space = 50
    # bilateral_blurred = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)

    # binarization_low = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                          cv2.THRESH_BINARY_INV,
    #                                          21, 11)
    #
    # cv2.namedWindow('bw', cv2.WINDOW_NORMAL)
    # cv2.imshow('bw', binarization_low)

    # median_blurred = cv2.medianBlur(gray, 5)
    # median_blurred_low = cv2.medianBlur(binarization_low, 3)

    # # Erosion
    # eroded_img_high = cv2.erode(gray, None)
    # eroded_img_low = cv2.erode(binarization_low, None)

    # dilated_img = cv2.dilate(binarization_low, None)
    # eroded_img = cv2.erode(dilated_img, None)

    # Opening morphology

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(binarization_low, cv2.MORPH_OPEN, kernel)

    # binarized_image.show()

    # cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
    # cv2.imshow('opening', opening)
    #
    # cv2.waitKey(0)

    return binarized_image


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
                                     False, (r, g, b), 4)

        cv2.namedWindow(f'{img}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'{img}', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cv2.namedWindow(f'{img}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'{img}', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Could not detect anything')

    return cv_image


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
        cv2.namedWindow(f'{cv_image}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'{cv_image}', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Could not detect anything')


def crop_image(input_image, lines):
    """
    :param input_image: original input image
    :param lines: coordinates of detected lines returned by Kraken line segmenter.
    :return: list containing all the cropped images
    """
    final_cropped_image = []
    for baseline in lines:
        base_line = baseline['boundary']
        base_line = np.array(base_line, np.int32)
        rect = cv2.boundingRect(base_line)
        x, y, w, h = rect
        cropped_image = input_image[y:y + h, x - 20:x + w + 20].copy()  # taking extra 20 pixels on both sides.
        # cv2.namedWindow('cropped image', cv2.WINDOW_NORMAL)
        # cv2.imshow('cropped image', cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # pts = base_line - base_line.min(axis=0)

        # mask = np.zeros(cropped_image.shape[:2], np.uint8)
        # cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # dst = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

        # bg = np.ones_like(cropped_image, np.uint8) * 255
        # cv2.bitwise_not(bg, bg, mask=mask)
        # dst2 = bg + dst
        # cv2.namedWindow('dst2', cv2.WINDOW_NORMAL)
        # cv2.imshow('dst2', dst2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        final_cropped_image.append(cropped_image)
    return final_cropped_image


if __name__ == "__main__":
    import os

    ########################## single image ###################################
    # image_path = '/home/dell/Documents/handwritten_images/testingimages/d3.jpg'
    #
    # cv_image = cv2.imread(image_path)
    # t = histogram_equalization_hsv(cv_image)
    # # image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # # V = image_hsv.copy()
    # # V[:, :, 0] = 179  # set H to max (179)
    # # V[:, :, 1] = 0  # set S to 0
    # # V_RGB = cv2.cvtColor(V, cv2.COLOR_HSV2RGB)  # convert back to RGB
    #
    # pil_image = Image.fromarray(t)  # reading with pil as a requirement of Kraken.
    # # pil_image.show()
    # baseline_seg = blla.segment(pil_image, model=None, device='cpu')  # Baseline segmenter
    # # print(baseline_seg['lines'])
    # # print(baseline_seg['lines']['baseline'])
    # # for i in baseline_seg['lines']:
    # #     print('value of baselines', i['baseline'])
    # #     print('value of boundary', i['boundary'])
    # #     print()
    #
    # visualize_polylines(image_path, cv_image, baseline_seg['lines'])
    # # cropped_images = crop_image(cv_image, baseline_seg['lines'])

    # for i in cropped_images:
    #     # img = cv2.imread(i)
    #     cv2.namedWindow(f'img', cv2.WINDOW_NORMAL)
    #     cv2.imshow(f'img', i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # draw_polygon(cv_image, baseline_seg['lines'])

    ######################################################################################################

    ####################################multiple image###################################################

    dir_path = '/home/dell/Documents/handwritten_images/testingimages'

    for img in sorted(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, img)
        print(image_path)

        # model_path = 'path/to/model/file'
        # model = vgsl.TorchVGSLModel.load_model(model_path)
        cv_image = cv2.imread(image_path)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        # image.show()
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # he = cv2.equalizeHist(cv_image)

        preprocessed = preprocessing(image)
        # pil_image = Image.fromarray(cv_image)  # reading with pil as a requirement of Kraken.
        # # pil_image.show()
        baseline_seg = blla.segment(preprocessed, model=None, device='cpu')  # Baseline segmenter

        visualize_polylines(img, cv_image, baseline_seg['lines'])
    # cropped_images = crop_image(cv_image, baseline_seg['lines'])
    #
    #     for i in cropped_images:
    #         print(i)
    #         # img = cv2.imread(i)
    #         cv2.namedWindow(f'{img}', cv2.WINDOW_NORMAL)
    #         cv2.imshow(f'{img}', i)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    # draw_polygon(cv_image, baseline_seg['lines'])
