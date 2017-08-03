from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import cv2
import numpy as np

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
            # result.append({'score': str(100 * score), 'label': human_string})
            # result.append('{0} {1}'.format(human_string, 100 * score))
            result.append((human_string, score))
        return result


# load labels
labels = load_labels('/tmp/output_labels.txt')

# load graph, which is stored in the default session
load_graph('/tmp/output_graph.pb')


IMAGE_WIDTH = 250
IMAGE_HEIGHT = 100
DETAIL = 1


class TensorFlowAPIHandler(object):
    def traverse(self, image_data):
        x_advance_rate = int(IMAGE_WIDTH / DETAIL)
        y_advance_rate = int(IMAGE_HEIGHT / DETAIL)
        height, width = image_data.shape
        y_steps_count = int(height / y_advance_rate)
        x_steps_count = int(width / x_advance_rate)
        for y0 in range(0, y_steps_count * y_advance_rate, y_advance_rate):
            for x0 in range(0, x_steps_count * x_advance_rate, x_advance_rate):
                partial = image_data[y0:y0 + IMAGE_HEIGHT, x0:x0 + IMAGE_WIDTH]
                yield partial

    def predict(self, image_data):
        np_array = np.fromstring(image_data, np.uint8)
        image_data = cv2.imdecode(np_array, 0)
        image_data = cv2.resize(image_data, (721, 1281), interpolation=cv2.INTER_CUBIC)
        return_value = []
        for window in self.traverse(image_data):
            _, buf = cv2.imencode('.jpg', window)
            window = np.array(buf).tostring()
            results = run_graph(window, labels, 'DecodeJpeg/contents:0', 'final_result:0', 3)
            return_value.append('{0}-{1}'.format(results[0][0], results[0][1]))
        return return_value


s = zerorpc.Server(TensorFlowAPIHandler())
s.bind("tcp://0.0.0.0:4242")
s.run()
