import os
import sys
import tarfile

from six.moves import urllib

import tensorflow as tf


def download_and_extract(model_info, dest_dir):
    """
    Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
        data_url: Web location of the tar file containing the pretrained model.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    file_name = os.path.basename(model_info.data_url)
    file_path = os.path.join(dest_dir, file_name)
    # if it doesn't exist already, download it.
    if not os.path.exists(file_path):
        def __progress(count, block_size, total_size):
            sys.stdout.write("\r[X] Downloading '{file_name}' {progress:.1f}%".format(
                    file_name=file_name,
                    progress=float(count * block_size) / float(total_size) * 100.0,
                )
            )

            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(model_info.data_url, file_path, __progress)
        print()
        file_stats = os.stat(file_path)
        tf.logging.info('Successfully downloaded {file_name} {size}'.format(
                file_name=file_name,
                size=file_stats.st_size,
            )
        )

    tarfile.open(file_path, 'r:gz').extractall(dest_dir)
