import os

import tensorflow as tf

from tensorflow.python.framework import tensor_shape


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """
    Adds operations that perform JPEG decoding and resizing to the graph.

    Args:
        input_width: Desired width of the image fed into the recognizer graph.
        input_height: Desired width of the image fed into the recognizer graph.
        input_depth: Desired channels of the image fed into the recognizer graph.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.

    Returns:
        Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def add_input_distortions(flip_left_right, random_scale, random_brightness, model_info):
    """
    Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
        flip_left_right: Boolean whether to randomly mirror images horizontally.
        random_crop: Integer percentage setting the total margin used around the
        crop box.
        random_scale: Integer percentage of how much to vary the scale by.
        random_brightness: Integer range to randomly multiply the pixel values by.
        graph.
        input_width: Horizontal size of expected input image to model.
        input_height: Vertical size of expected input image to model.
        input_depth: How many channels the expected input image should have.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.

    Returns:
        The jpeg input layer and the distorted result tensor.
    """
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=model_info.input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_scale = 1.0 + (random_scale / 100.0)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
    precrop_width = tf.multiply(resize_scale_value, model_info.input_width)
    precrop_height = tf.multiply(resize_scale_value, model_info.input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    decoded_image = tf.random_crop(
        precropped_image_3d,
        [
            model_info.input_height,
            model_info.input_width,
            model_info.input_depth,
        ]
    )
    if flip_left_right:
        decoded_image = tf.image.random_flip_left_right(decoded_image)
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(decoded_image, brightness_value)
    offset_image = tf.subtract(brightened_image, model_info.input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / model_info.input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result
