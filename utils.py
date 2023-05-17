import os


def rename_move_image(source_dir_path, destination_dir_path):
    """
    For image pre processing
    :param source_dir_path:
    :param destination_dir_path:
    :return:
    """
    s = 3  # new name
    sufix = 'd'  # if all the images' name should have a common Sufix.

    for image in os.listdir(source_dir_path):
        image_path = os.path.join(source_dir_path, image)
        extensor = image.split('.')[-1]
        print('Image name:', image)
        new_image_name = f'{sufix}' + str(s) + '.' + extensor
        print('New Image Name:', new_image_name)
        new_path = os.path.join(destination_dir_path, new_image_name)
        print(new_path)
        print("*****")
        os.rename(image_path, new_path)
        s += 1


if __name__ == "__main__":
    source_dir_path = '/home/dell/Documents/handwritten_images/notespicture'
    destination_dir_path = '/home/dell/Documents/handwritten_images/testingimages/'
