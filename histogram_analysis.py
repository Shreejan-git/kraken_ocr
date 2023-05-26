import os
import cv2
import matplotlib.pyplot as plt


def histogram_analysis(image_path, saving_dir_path='/'):
    """
    saving the image histogram
    :param image_path:
    :return:
    """
    image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1]
    saving_dir_path = os.path.join(saving_dir_path, image_name)

    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Compute the histograms for each color channel
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Plot the histogramsq
    plt.figure(figsize=(8, 6))
    plt.plot(hist_b, color='blue', label='Blue')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_r, color='red', label='Red')
    plt.title(f'Color Histogram of {image_name}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(saving_dir_path) # Uncomment to save the figure
    # plt.show() # Uncomment to show the figure
    plt.close()


if __name__ == "__main__":
    # Read the color image
    # image_path = '/home/dell/Documents/handwritten_images/testingimages/d1.jpeg'
    dir_path = '/home/dell/Documents/handwritten_images/new_test_cases'
    saving_dir_path = '/home/dell/Documents/handwritten_images/new_histogram_analysis'
    for img in sorted(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, img)
        histogram_analysis(image_path, saving_dir_path)



