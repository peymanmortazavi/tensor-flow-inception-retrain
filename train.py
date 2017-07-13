import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
  '--training_dir',
  type=str,
  default=os.environ.get('TRAINING_DIR') or '',
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
  '--model_path',
  type=str,
  default=os.environ.get('MODEL_PATH') or '/tmp/imagenet',
  help='Path to classify_image_graph_def.pb'
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
