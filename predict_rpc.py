from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import zerorpc


def load_image(filename):
    """
    Read in the image_data to be classified.
    """
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
    """
    Read in labels, one label per line.
    """
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    """
    Unpersists graph from file as default graph.
    """
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        result = []
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            result.append({'score': score, 'label': human_string})


# load labels
labels = load_labels('/tmp/output_labels.txt')

# load graph, which is stored in the default session
load_graph('/tmp/output_graph.pb')


class TensorFlowAPIHandler(object):
    def predict(self, image_data):
        return run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 3)


s = zerorpc.Server(TensorFlowAPIHandler())
s.bind("tcp://0.0.0.0:4242")
s.run()
