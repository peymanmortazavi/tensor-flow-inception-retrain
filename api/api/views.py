from django.http.response import HttpResponse

import os
import tensorflow as tf
import cv2
import numpy as np
import json
import urllib2
from preprocessing.crop_vertical import crop_vertical


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
            result.append((human_string, score))
        return result


# load labels
labels = load_labels(os.environ.get('MODEL_LABELS'))

# load graph, which is stored in the default session
load_graph(os.environ.get('MODEL_GRAPH'))


def predict(request):
    try:
        response = urllib2.urlopen(request.GET.get('image_url'))
        image_data = response.read()
        np_array = np.fromstring(image_data, np.uint8)
        image_data = cv2.imdecode(np_array, 0)
        image_data = cv2.resize(image_data, (721, 1281), interpolation=cv2.INTER_CUBIC)
        cropped_image_data = crop_vertical(image_data, 0.5, (299, 299), 0.5)
        _, buff = cv2.imencode('.jpg', cropped_image_data)
        best_result = run_graph(np.array(buff).tostring(), labels, 'DecodeJpeg/contents:0', 'final_result:0', 3)[0]
        result_dict = {'label': best_result[0], 'conf': str(best_result[1] * 100)}
        return HttpResponse(json.dumps(result_dict))
    except Exception, e:
        return HttpResponse(str(e), status=400)