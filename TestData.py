from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

##################load data#####################

all_data = pickle.load(open('CIFAR_100_normalized.pickle', 'rb'))

test_data = all_data['test_dataset']
test_labels = all_data['test_labels']

del all_data

num_channels = 1  # grayscale
image_size = 32


def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset

test_data = reformat(test_data)

test_size = test_data.shape[0]

test_batch_size = 50  # used to calculate test predictions over many iterations to avoid memory issues
images_labels = 100  # the labels' length for the classifier


# computes accuracy given the predictions and real labels
def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc, predictions


with tf.Session() as sess:
    model_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')
    model_saver.restore(sess, tf.train.latest_checkpoint('./best_model/saved_model/'))
    graph = sess.graph
    inputs = graph.get_tensor_by_name("tf_inputs:0")
    keep_prob = graph.get_tensor_by_name("fully_connected_keep_prob:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    tf_predictions = graph.get_tensor_by_name("tf_predictions:0")

    # print([node.name for node in graph.as_graph_def().node])

    # get test predictions in steps to avoid memory problems

    test_pred = np.zeros((test_size, images_labels))
    for step in range(int(test_size / test_batch_size)):
        offset = (step * test_batch_size) % (test_size - test_batch_size)
        batch_data = test_data[offset:(offset + test_batch_size), :]
        feed_dict = {inputs: batch_data, keep_prob: 1.0, is_training: False}
        predictions = sess.run(
            tf_predictions, feed_dict=feed_dict)

        test_pred[offset:offset + test_batch_size, :] = predictions

    # calculate test accuracy and save the model

    test_accuracy, test_predictions = accuracy(np.argmax(test_pred, axis=1), test_labels)

    print('Test accuracy: %.1f%%' % test_accuracy)
