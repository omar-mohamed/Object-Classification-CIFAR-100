from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from matplotlib import pyplot as plt
import random

# from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

##################load data#####################

all_data = pickle.load(open('CIFAR_100_normalized.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']
valid_data = all_data['valid_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']
valid_labels = all_data['valid_labels']
label_names = all_data['label_names']

del all_data

#################Format train and test data###################


num_channels = 1  # grayscale
image_size = 32


def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


train_data = reformat(train_data)
test_data = reformat(test_data)
valid_data = reformat(valid_data)

print('train_data shape is : %s' % (train_data.shape,))
print('test_data shape is : %s' % (test_data.shape,))
print('valid_data shape is : %s' % (valid_data.shape,))

test_size = test_data.shape[0]
validation_size = valid_data.shape[0]
train_size = train_data.shape[0]


############################################################



########################Training###########################



# computes accuracy given the predictions and real labels
def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc, predictions


# output width=((W-F+2*P )/S)+1



images_labels = 100  # the labels' length for the classifier
batch_size = 20  # the number of training samples in a single iteration
test_batch_size = 50  # used to calculate test predictions over many iterations to avoid memory issues
patch_size_1 = 7  # convolution filter size 1
patch_size_2 = 5  # convolution filter size 2
patch_size_3 = 3  # convolution filter size 3
patch_size_4 = 1  # convolution filter size 4

depth1 = 32  # number of filters in first conv layer
depth2 = 64  # number of filters in second conv layer
depth3 = 128  # number of filters in third conv layer
depth4 = 256  # number of filters in first conv layer

num_hidden1 = 1024  # the size of the unrolled vector after convolution
num_hidden2 = 128  # the size of the hidden neurons in fully connected layer
num_hidden3 = 1024  # the size of the hidden neurons in fully connected layer
regularization_lambda = 4e-2  # used in case of L2 regularization

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_inputs = tf.placeholder(
        tf.float32, shape=(None, image_size, image_size, num_channels), name='tf_inputs')

    # label for each image
    tf_labels = tf.placeholder(tf.int32, shape=None, name='tf_labels')

    fully_connected_keep_prob = tf.placeholder(tf.float32, name='fully_connected_keep_prob')

    is_training = tf.placeholder(tf.bool, name='is_training')


    def get_conv_weight(name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())


    def get_bias_variable(name, shape):
        return tf.Variable(tf.constant(1.0, shape=shape), name=name)


    def get_fully_connected_weight(name, shape):
        weights = tf.get_variable(name, shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        return weights


    # Convolution Variables.

    conv1_weights = get_conv_weight('conv1_weights', [patch_size_1, patch_size_1, num_channels, depth1])

    conv2_weights = get_conv_weight('conv2_weights', [patch_size_2, patch_size_2, depth1, depth2])

    conv3_weights = get_conv_weight('conv3_weights', [patch_size_3, patch_size_3, depth2, depth3])

    conv4_weights = get_conv_weight('conv4_weights', [patch_size_4, patch_size_4, depth3, depth4])

    # Fully connected Variables.

    hidden1_weights_c1 = get_fully_connected_weight('hidden1_weights', [num_hidden1, num_hidden2])

    hidden2_weights_c1 = get_fully_connected_weight('hidden2_weights', [num_hidden2, num_hidden3])

    hidden3_weights_c1 = get_fully_connected_weight('hidden3_weights', [num_hidden3, images_labels])


    # method that runs one hidden layer with batch normalization and dropout
    def run_hidden_layer(x, hidden_weights, keep_dropout_rate=1, use_relu=True, is_training=False):
        hidden = tf.matmul(x, hidden_weights)

        hidden = tf.layers.batch_normalization(
            inputs=hidden,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training
        )

        if use_relu:
            hidden = tf.nn.relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_dropout_rate)
        return hidden


    # method that runs one convolution layer with batch normalization
    def run_conv_layer(x, conv_weights):
        conv = tf.nn.conv2d(x, conv_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.layers.batch_normalization(
            inputs=conv,
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training
        )
        return tf.nn.relu(conv)


    # Model.
    def model(data, keep_dropout_rate=1):
        hidden = data
        # first conv block
        hidden = run_conv_layer(hidden, conv1_weights)
        # second conv block
        hidden = run_conv_layer(hidden, conv2_weights)
        # second conv block
        hidden = run_conv_layer(hidden, conv3_weights)
        # second conv block
        hidden = run_conv_layer(hidden, conv4_weights)
        # flatten
        hidden = tf.contrib.layers.flatten(hidden)

        #  classifier
        # hidden = run_hidden_layer(hidden, hidden1_weights_c1, keep_dropout_rate, True)
        #
        # hidden = run_hidden_layer(hidden, hidden2_weights_c1, keep_dropout_rate, True)

        hidden = run_hidden_layer(hidden, hidden3_weights_c1, 1, False)

        return hidden


    # Training computation.
    logits = model(tf_inputs, fully_connected_keep_prob)

    # loss of softmax with cross entropy

    tf_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits))

    # for saving batch normalization values
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # use learning rate decay

        # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
        # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.0001, global_step, 10000, 0.50, staircase=True)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(tf_loss, tvars),
                                          100.0)  # gradient clipping
        optimize = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=global_step)

    # Predictions for the inputs.

    tf_predictions = tf.nn.softmax(logits)
    tf_predictions = tf.identity(tf_predictions, name='tf_predictions')

########################Training Session###########################


num_steps = 50001  # number of training iterations

# used for drawing error and accuracy over time
training_loss = []
training_loss_iteration = []

valid_loss = []
valid_loss_iteration = []

train_accuracy = []
train_accuracy_iteration = []

valid_accuracy = []
valid_accuracy_iteration = []

test_accuracy = 0

print("Training CNN")

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.global_variables_initializer().run()
    # to save model after finishing
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graph_info', session.graph)

    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_size - batch_size)

        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size)]

        # train on batch and get accuracy and loss
        feed_dict = {tf_inputs: batch_data, tf_labels: batch_labels, fully_connected_keep_prob: 0.75, is_training: True}
        _, l, predictions, lr = session.run(
            [optimize, tf_loss, tf_predictions, learning_rate], feed_dict=feed_dict)

        if (step % 50 == 0):
            print('Learning rate at step %d: %.14f' % (step, lr))
            print('Minibatch loss at step %d: %f' % (step, l))
            batch_train_accuracy, _ = accuracy(np.argmax(predictions, axis=1), batch_labels)
            print('Minibatch accuracy: %.1f%%' % batch_train_accuracy)
            training_loss.append(l)
            training_loss_iteration.append(step)
            train_accuracy.append(batch_train_accuracy)
            train_accuracy_iteration.append(step)

        if (step % 500 == 0):
            feed_dict = {tf_inputs: valid_data, tf_labels: valid_labels, fully_connected_keep_prob: 1.0,
                         is_training: False}
            l, predictions = session.run(
                [tf_loss, tf_predictions], feed_dict=feed_dict)
            print('validation set loss at step %d: %f' % (step, l))
            acc, _ = accuracy(np.argmax(predictions, axis=1), valid_labels)
            print('validation set accuracy: %.1f%%' % acc)
            valid_loss.append(l)
            valid_loss_iteration.append(step)
            valid_accuracy.append(acc)
            valid_accuracy_iteration.append(step)

    # get test predictions in steps to avoid memory problems

    test_pred = np.zeros((test_size, images_labels))

    for step in range(int(test_size / test_batch_size)):
        offset = (step * test_batch_size) % (test_size - test_batch_size)
        batch_data = test_data[offset:(offset + test_batch_size), :]
        feed_dict = {tf_inputs: batch_data, fully_connected_keep_prob: 1.0, is_training: False}
        predictions = session.run(
            tf_predictions, feed_dict=feed_dict)

        test_pred[offset:offset + test_batch_size, :] = predictions

    # calculate test accuracy and save the model

    test_accuracy, test_predictions = accuracy(np.argmax(test_pred, axis=1), test_labels)
    writer.close()
    saver.save(session, "./saved_model/model.ckpt")


###############################Plot Results and save images##############################

# saves accuracy and loss images in folder output_images
def plot_x_y(x, y, figure_name, x_axis_name, y_axis_name, ylim=[0, 100]):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    axes = plt.gca()
    axes.set_ylim(ylim)
    # plt.legend([line_name],loc='upper left')
    plt.savefig('./output_images/' + figure_name)
    # plt.show()


plot_x_y(training_loss_iteration, training_loss, 'training_loss.png', 'iteration', 'training batch loss', [0, 15])

plot_x_y(train_accuracy_iteration, train_accuracy, 'training_acc.png', 'iteration', 'training batch accuracy')

plot_x_y(valid_loss_iteration, valid_loss, 'valid_loss.png', 'iteration', 'valid loss', [0, 15])

plot_x_y(valid_accuracy_iteration, valid_accuracy, 'valid_acc.png', 'iteration', 'validation accuracy')


def disp_prediction_samples(predictions, dataset, num_images, cmap=None):
    for image_num in range(num_images):
        items = random.sample(range(dataset.shape[0]), 8)
        for i, item in enumerate(items):
            plt.subplot(2, 4, i + 1)
            plt.axis('off')
            plt.title(label_names[predictions[i]])
            plt.imshow(dataset[item, :, :, 0], cmap=cmap)
        plt.savefig('./output_images/' + 'predictions' + str(image_num + 1) + '.png')
        # plt.show()


disp_prediction_samples(test_predictions, test_data, 10)

print('Test accuracy: %.1f%%' % test_accuracy)
