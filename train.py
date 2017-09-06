import argparse
import datetime
import os
import sys

from bottleneck import cache_bottlenecks, get_random_cached_bottlenecks, get_random_distorted_bottlenecks

from image_list import create_image_lists

import model

import tensorflow as tf
from tensorflow.python.platform import gfile

from utils import add_input_distortions, add_jpeg_decoding


parser = argparse.ArgumentParser()
parser.add_argument(
  '--image_dir',
  type=str,
  default=os.environ.get('IMAGE_DIR') or '',
  help='Path to directory containing the training image data.'
)
parser.add_argument(
  '--output_graph',
  type=str,
  default=os.environ.get('OUTPUT_GRAPH') or '/tmp/output_graph.pb',
  help='Where to save the trained graph.'
)
parser.add_argument(
  '--output_labels',
  type=str,
  default=os.environ.get('OUTPUT_LABELS') or '/tmp/output_labels.txt',
  help='Where to save the trained graph\'s labels.'
)
parser.add_argument(
  '--summaries_dir',
  type=str,
  default=os.environ.get('OUTPUT_SUMMARIES') or '/tmp/retrain_logs',
  help='Where to save summary logs for TensorBoard.'
)
parser.add_argument(
  '--steps_count',
  type=int,
  default=int(os.environ.get('STEPS_COUNT', 0)) or 4000,
  help='How many training steps to run before ending.'
)
parser.add_argument(
  '--learning_rate',
  type=float,
  default=float(os.environ.get('LEARNING_RATE', 0)) or 0.01,
  help='How large a learning rate to use when training.'
)
parser.add_argument(
  '--testing_percentage',
  type=int,
  default=int(os.environ.get('TESTING_PERCENTAGE', 0)) or 10,
  help='What percentage of images to use as a test set.'
)
parser.add_argument(
  '--validation_percentage',
  type=int,
  default=int(os.environ.get('VALIDATION_PERCENTAGE', 0)) or 10,
  help='What percentage of images to use as a validation set.'
)
parser.add_argument(
  '--eval_step_interval',
  type=int,
  default=int(os.environ.get('EVAL_STEP_INTERVAL', 0)) or 10,
  help='How often to evaluate the training results.'
)
parser.add_argument(
  '--train_batch_size',
  type=int,
  default=int(os.environ.get('TRAIN_BATCH_SIZE', 0)) or 100,
  help='How many images to train on at a time.'
)
parser.add_argument(
  '--test_batch_size',
  type=int,
  default=int(os.environ.get('TEST_BATCH_SIZE', 0)) or -1,
  help="""\
  How many images to test on. This test set is only used once, to evaluate
  the final accuracy of the model after training completes.
  A value of -1 causes the entire test set to be used, which leads to more
  stable results across runs.\
  """
)
parser.add_argument(
  '--validation_batch_size',
  type=int,
  default=int(os.environ.get('VALIDATION_BATCH_SIZE', 0)) or 100,
  help="""\
  How many images to use in an evaluation batch. This validation set is
  used much more often than the test set, and is an early indicator of how
  accurate the model is during training.
  A value of -1 causes the entire validation set to be used, which leads to
  more stable results across training iterations, but may be slower on large
  training sets.\
  """
)
parser.add_argument(
  '--print_misclassified_test_images',
  default=False,
  help="""\
  Whether to print out a list of all misclassified test images.\
  """,
  action='store_true'
)
parser.add_argument(
  '--model_dir',
  type=str,
  default=os.environ.get('MODEL_DIR') or '/tmp/imagenet',
  help='Directory in which graph definitions (*.pb) live.'
)
parser.add_argument(
  '--bottleneck_dir',
  type=str,
  default=os.environ.get('BOTTLENECK_DIR') or '/tmp/bottleneck',
  help='Path to cache bottleneck layer values as files.'
)
parser.add_argument(
  '--final_tensor_name',
  type=str,
  default=os.environ.get('FINAL_TENSOR_NAME') or 'final_result',
  help="""\
  The name of the output classification layer in the retrained graph.\
  """
)
parser.add_argument(
  '--flip_left_right',
  default=os.environ.get('ENABLE_HORIZONTAL_FLIP') == 'yes',
  help="""\
  Whether to randomly flip half of the training images horizontally.\
  """,
  action='store_true'
)
parser.add_argument(
  '--random_scale',
  type=int,
  default=int(os.environ.get('RANDOM_SCALE_PERCENTAGE', 0)) or 0,
  help="""\
  A percentage determining how much to randomly scale up the size of the
  training images by.\
  """
)
parser.add_argument(
  '--random_brightness',
  type=int,
  default=int(os.environ.get('RANDOM_BRIGHTNESS_PERCENTAGE', 0)) or 0,
  help="""\
  A percentage determining how much to randomly multiply the training image
  input pixels up or down by.\
  """
)


FLAGS, unparsed = parser.parse_known_args()


def variable_summaries(variable):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)


def add_final_training_ops(
            class_count,
            final_tensor_name,
            bottleneck_tensor,
            bottleneck_tensor_size
        ):
    """
    Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
        class_count: Integer of how many categories of things we're trying to
        recognize.
        final_tensor_name: Name string for the new final node that produces results.
        bottleneck_tensor: The output of the main CNN graph.
        bottleneck_tensor_size: How many entries in the bottleneck vector.

    Returns:
        The tensors for the training and cross entropy results, and tensors for the
        bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder',
        )

        ground_truth_input = tf.placeholder(
            tf.float32,
            [None, class_count],
            name='GroundTruthInput',
        )

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                shape=[bottleneck_tensor_size, class_count],
                stddev=0.001
            )

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input,
            logits=logits,
        )

        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (
        train_step,
        cross_entropy_mean,
        bottleneck_input,
        ground_truth_input,
        final_tensor
    )


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

    Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))

        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Make sure we have the summaries directory setup.
    if gfile.Exists(FLAGS.summaries_dir):
        gfile.DeleteRecursively(FLAGS.summaries_dir)
    gfile.MakeDirs(FLAGS.summaries_dir)

    # Get the info of the model we'll be using for this training.
    model_info = model.pretrained.Inception.get_model_info()

    # Download and extract the model so it's available locally.
    # Then setup the pre-trained graph
    model.download_and_extract(model_info, dest_dir=FLAGS.model_dir)
    graph, bottleneck_tensor, resized_image_tensor = model.create_model_graph(model_info, model_dir=FLAGS.model_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    if not image_lists:
        tf.logging.error("No valid folders of images found at '{image_dir}'".format(image_dir=FLAGS.image_dir))
        return -1
    if len(image_lists) == 1:
        tf.logging.error(
            """Only one valid folder of images found at '{image_dir}'"""
            """ - multiple classes are needed for classification.""".format(
                image_dir=FLAGS.image_dir
            )
        )
        return -1

    # Now see if any distortions need to get applied.
    do_distort_images = FLAGS.flip_left_right or FLAGS.random_scale or FLAGS.random_brightness

    with tf.Session(graph=graph) as session:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info.input_width,
            model_info.input_height,
            model_info.input_depth,
            model_info.input_mean,
            model_info.input_std,
        )

        if do_distort_images:
            # We will be applying distortions, so setup the operations we'll need.
            distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
               FLAGS.flip_left_right,
               FLAGS.random_scale,
               FLAGS.random_brightness,
               model_info,
            )
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(
                session,
                image_lists,
                FLAGS.image_dir,
                FLAGS.bottleneck_dir,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_image_tensor,
                bottleneck_tensor,
            )

        # Add the new layer that we'll be training.
        train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor = add_final_training_ops(
             len(image_lists),
             FLAGS.final_tensor_name,
             bottleneck_tensor,
             model_info.bottleneck_tensor_size
        )

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'train'), session.graph)

        validation_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'validation'))

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        session.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.steps_count):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                    session,
                    image_lists,
                    FLAGS.train_batch_size,
                    'training',
                    FLAGS.image_dir,
                    distorted_jpeg_data_tensor,
                    distorted_image_tensor,
                    resized_image_tensor,
                    bottleneck_tensor
                )
            else:
                train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                    session,
                    image_lists,
                    FLAGS.train_batch_size,
                    'training',
                    FLAGS.bottleneck_dir,
                    FLAGS.image_dir,
                    jpeg_data_tensor,
                    decoded_image_tensor,
                    resized_image_tensor,
                    bottleneck_tensor
                )

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = session.run(
                [merged, train_step],
                feed_dict={
                    bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth
                }
            )
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.steps_count)
            is_eval_step = (i % FLAGS.eval_step_interval == 0)
            if is_eval_step or is_last_step:
                train_accuracy, cross_entropy_value = session.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth,
                    },
                )
                tf.logging.info("{now}: Step {step_num}: Train accuracy = {accuracy:.1f}%".format(
                        now=datetime.datetime.now(),
                        step_num=i,
                        accuracy=train_accuracy * 100,
                    )
                )
                tf.logging.info("{now}: Step {step_num}: Cross entropy = {entropy}".format(
                        now=datetime.datetime.now(),
                        step_num=i,
                        entropy=cross_entropy_value,
                    )
                )
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        session,
                        image_lists,
                        FLAGS.validation_batch_size,
                        'validation',
                        FLAGS.bottleneck_dir,
                        FLAGS.image_dir,
                        jpeg_data_tensor,
                        decoded_image_tensor,
                        resized_image_tensor,
                        bottleneck_tensor,
                    )
                )
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = session.run(
                    [merged, evaluation_step],
                    feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input: validation_ground_truth,
                    },
                )
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info(
                    "{now}: Step {step_num}: Validation accuracy = {accuracy:.1f}% (N={num_bottleneck})".format(
                        now=datetime.datetime.now(),
                        step_num=i,
                        accuracy=validation_accuracy * 100,
                        num_bottleneck=len(validation_bottlenecks),
                    )
                )

            if i % 500 == 0 and i > 0:
                intermediate_graph_path = os.path.join(os.path.dirname(FLAGS.output_graph), 'graph-{}.pb'.format(i))
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_graph_path)
                model.save_graph_to_file(session, graph, intermediate_graph_path, FLAGS.final_tensor_name)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(
                session,
                image_lists,
                FLAGS.test_batch_size,
                'testing',
                FLAGS.bottleneck_dir,
                FLAGS.image_dir,
                jpeg_data_tensor,
                decoded_image_tensor,
                resized_image_tensor,
                bottleneck_tensor
            )
        )
        test_accuracy, predictions = session.run(
            [evaluation_step, prediction],
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input: test_ground_truth,
            }
        )
        tf.logging.info("Final test accuracy = {accuracy:.1f}% (N={bottleneck_count})".format(
                accuracy=test_accuracy * 100,
                bottleneck_count=len(test_bottlenecks)
            )
        )

        if FLAGS.print_misclassified_test_images:
            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    tf.logging_info('{filename} - {predictions}'.format(
                            filename=test_filename,
                            predictions=list(image_lists.keys())[predictions[i]],
                        )
                    )

        # Write out the trained graph and labels with the weights stored as
        # constants.
        model.save_graph_to_file(session, graph, FLAGS.output_graph, FLAGS.final_tensor_name)
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')


tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
