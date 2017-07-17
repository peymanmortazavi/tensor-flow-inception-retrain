import os
import random

import constants

import numpy as np

import tensorflow as tf

from tensorflow.python.platform import gfile

from utils import ensure_dir_exists


def get_image_path(image_lists, label_name, index, image_dir, category):
    """
    Returns a path to an image for a label at the given index.

    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Int offset of the image we want. This will be moduloed by the
        available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string of the subfolders containing the training
        images.
        category: Name string of set to pull images from - training, testing, or
        validation.

    Returns:
        File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        tf.logging.fatal("Label does not exist '{label}'".format(label=label_name))
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal(
            "Label '{label}' has no image in the category '{category}'.".format(
                label=label_name,
                category=category,
            )
        )

    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['label']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(
            image_lists,
            label_name,
            index,
            bottleneck_dir,
            category
        ):
    """
    Returns a path to a bottleneck file for a label at the given index.

    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be moduloed by the
        available number of images for the label, so it can be arbitrarily large.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        category: Name string of set to pull images from - training, testing, or
        validation.

    Returns:
        File system path string to an image that meets the requested parameters.
    """
    image_path = get_image_path(
        image_lists,
        label_name,
        index,
        bottleneck_dir,
        category
    )
    return "{image_path}.txt".format(image_path=image_path)


def run_bottleneck_on_image(
            session,
            image_data,
            image_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """
    Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
        sess: Current active TensorFlow Session.
        image_data: String of raw JPEG data.
        image_data_tensor: Input data layer in the graph.
        decoded_image_tensor: Output of initial image resizing and  preprocessing.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: Layer before the final softmax.

    Returns:
        Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = session.run(decoded_image_tensor, {image_data_tensor: image_data})

    # Then run it through the recognition network.
    bottleneck_values = session.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def create_bottleneck_file(
            bottleneck_path,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            session,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """
    Create a single bottleneck file.
    """
    tf.logging.info("Creating bottleneck at '{path}'".format(path=bottleneck_path))
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal("File does not exist '{image_path}'".format(image_path=image_path))
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            session,
            image_data,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor,
        )
    except Exception as e:
        raise RuntimeError(
            "Error occurred during processing file '{image_path}': {error}".format(
                image_path=image_path,
                error=str(e)
            )
        )

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(
            session,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            bottleneck_dir,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be modulo-ed by the
        available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string  of the subfolders containing the training
        images.
        category: Name string of which  set to pull images from - training, testing,
        or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The output tensor for the bottleneck values.

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['label']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists,
        label_name,
        index,
        bottleneck_dir,
        category
    )

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(
            bottleneck_path,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            session,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        )

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False

    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True

    if did_hit_error:
        create_bottleneck_file(
            bottleneck_path,
            image_lists,
            label_name,
            index,
            image_dir,
            category,
            session,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor,
        )

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            # Allow exceptions to propagate here, since they shouldn't happen after a
            # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(
            session,
            image_lists,
            image_dir,
            bottleneck_dir,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """
    Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        image_dir: Root folder string of the subfolders containing the training
        images.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The penultimate output layer of the graph.

    Returns:
        Nothing.
    """
    bottleneck_count = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    session,
                    image_lists,
                    label_name,
                    index,
                    image_dir,
                    category,
                    bottleneck_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_input_tensor,
                    bottleneck_tensor,
                )

                bottleneck_count += 1
                if bottleneck_count % 100 == 0:
                    tf.logging.info(
                        "{bottleneck_count} bottleneck files created.".format(bottleneck_count=bottleneck_count)
                    )


def get_random_distorted_bottlenecks(
            session,
            image_lists,
            how_many,
            category,
            image_dir,
            input_jpeg_tensor,
            distorted_image,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """
    Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

    Args:
        session: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: The integer number of bottleneck values to return.
        category: Name string of which set of images to fetch - training, testing,
        or validation.
        image_dir: Root folder string of the subfolders containing the training
        images.
        input_jpeg_tensor: The input layer we feed the image data to.
        distorted_image: The output node of the distortion graph.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
        List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(constants.MAX_TRAINING_SET_SIZE + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)

        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = session.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = session.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck_values)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def get_random_cached_bottlenecks(
            session,
            image_lists,
            how_many,
            category,
            bottleneck_dir,
            image_dir,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
    """
    Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
        session: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: If positive, a random sample of this size will be chosen.
        If negative, all bottlenecks will be retrieved.
        category: Name string of which set to pull from - training, testing, or
        validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        image_dir: Root folder string of the subfolders containing the training
        images.
        jpeg_data_tensor: The layer to feed jpeg image data into.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
        List of bottleneck arrays, their corresponding ground truths, and the
        relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(constants.MAX_TRAINING_SET_SIZE + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)

            bottleneck = get_or_create_bottleneck(
                session,
                image_lists,
                label_name,
                image_index,
                image_dir,
                category,
                bottleneck_dir,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_input_tensor,
                bottleneck_tensor
            )
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    session,
                    image_lists,
                    label_name,
                    image_index,
                    image_dir,
                    category,
                    bottleneck_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_input_tensor,
                    bottleneck_tensor
                )
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames
