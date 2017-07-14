import os

import tensorflow as tf


def create_model_graph(model_info, model_dir):
    """
    Creates a graph from saved GraphDef file and returns a Graph object.

    Args:
        model_info: Dictionary containing information about the model architecture.

    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_info.model_file_name)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = tf.import_graph_def(
                graph_def=graph_def,
                name='',
                return_elements=[
                    model_info.bottleneck_tensor_name,
                    model_info.resized_input_tensor_name,
                ]
            )
    return graph, bottleneck_tensor, resized_input_tensor
