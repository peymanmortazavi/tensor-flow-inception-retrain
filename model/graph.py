import os

import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


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


def save_graph_to_file(session, graph, graph_file_name, tensor_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        session,
        graph.as_graph_def(),
        [tensor_name],
    )
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
