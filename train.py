import hashlib
import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile


ACCEPTED_EXTENSIONS = ('jpg', 'JPG', 'jpeg', 'JPEG')
MIN_TRAINING_SET_SIZE = 30
MAX_TRAINING_SET_SIZE = 2 ** 14 - 1  # 16383


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
        image_dir: String path to a folder containing subfolders of images.
        testing_percentage: Integer percentage of the images to reserve for tests.
        validation_percentage: Integer percentage of images reserved for validation.

    Returns:
        A dictionary containing an entry for each label subfolder, with images split
        into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '{image_dir}' does not exist.".format(image_dir=image_dir))
        return

    result = {}
    sub_dirs = [info[0] for info in gfile.Walk(image_dir)]
    for sub_dir in sub_dirs[1:]:  # first one is root, so skip it.
        dir_name = os.path.basename(sub_dir)
        tf.logging.info("Looking for images in '{dir_name}'".format(dir_name=dir_name))
        file_list = []
        for extension in ACCEPTED_EXTENSIONS:
            pattern = os.path.join(sub_dir, '*.{}'.format(extension))
            file_list.extend(gfile.Glob(pattern))

        # make sure we have enough
        if len(file_list) < MIN_TRAINING_SET_SIZE:
            tf.logging.warning(
                "Directory '{dir_name}' has less than {threshold} images, which may cause issues.".format(
                    dir_name=dir_name,
                    threshold=MIN_TRAINING_SET_SIZE,
                )
            )
        elif len(file_list) > MAX_TRAINING_SET_SIZE:
            tf.logging.warning(
                "Directory '{dir_name}' has too many images, only {threshold} images will get used.".format(
                    dir_name=dir_name,
                    threshold=MAX_TRAINING_SET_SIZE,
                )
            )
        # remove any character that is not alpha or number.
        label_name = re.sub(r'[^a-z0-9]', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            # Ignore anything that comes after _nohash_ so "walmart1_nohash_qr.jpg" will turn into "walmart1.jpg"
            # This is a way for the sampler (the person or script who collected the training data) to push same type of
            # image into the same set (because the hash function would yield the same number).
            hash_name = re.sub(r'_nohash_.*$', '', file_path)
            # This is a magic code that would determine which set this particular picture should go in.
            hash_name_hashed = hashlib.sha1(bytes(hash_name, encoding='utf8')).hexdigest()
            percentage_hash = (
                (int(hash_name_hashed, 16) % (MAX_TRAINING_SET_SIZE + 1)) *
                (100.0 / MAX_TRAINING_SET_SIZE)
            )
            if percentage_hash < validation_percentage:
                validation_images.append(file_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(file_name)
            else:
                training_images.append(file_name)
        result[label_name] = {
            'label': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


print(create_image_lists('./data/training/', 10, 10))
