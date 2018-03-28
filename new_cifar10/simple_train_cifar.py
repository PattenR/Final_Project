############################################################
#                                                          #
#  Code for Lab 1: Intro to TensorFlow and Blue Crystal 4  #
#                                                          #
############################################################

'''Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import os.path

import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
import cifar10 as cf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('datadir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('maxsteps', 100000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('logfrequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('savemodel', 100,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batchsize', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learningrate', 0.1, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('imgwidth', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('imgheight', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('imgchannels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('numclasses', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('logdir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')


run_log_dir = os.path.join(FLAGS.logdir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batchsize,
                                                        lr=FLAGS.learningrate))
def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

  Args:
      x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a standard CIFAR10 image.

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
      img_summary: a string tensor containing sampled input images.
    """
    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.

    x_image = tf.reshape(x, [-1, FLAGS.imgwidth, FLAGS.imgheight, FLAGS.imgchannels])

    img_summary = tf.summary.image('Input_images', x_image)

    # First convolutional layer - maps one image to 64 feature maps.
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, FLAGS.imgchannels, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram("weights", W_conv1)
        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)
        # Normalisation layer - helps generalisation
        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('Conv_2'):
        # Second convolutional layer -- maps 64 feature maps to 64.
        W_conv2 = weight_variable([5, 5, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
        tf.summary.histogram("weights", W_conv2)
        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)
        # Second normalisation layer
        norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.variable_scope('Conv_3'):
        # Third convolutional layer -- maps 64 feature maps to 64.
        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(norm2, W_conv3) + b_conv3)
        tf.summary.histogram("weights", W_conv3)
        # Third pooling layer.
        h_pool3 = max_pool_2x2(h_conv3)
        # Third normalisation layer
        norm3 = tf.nn.lrn(h_pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    with tf.variable_scope('Conv_4'):
        # Fourth convolutional layer -- maps 64 feature maps to 64.
        W_conv4 = weight_variable([5, 5, 64, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.nn.relu(conv2d(norm3, W_conv4) + b_conv4)
        tf.summary.histogram("weights", W_conv4)
        # Fourth pooling layer.
        h_pool4 = max_pool_2x2(h_conv4)
        # Fourth normalisation layer
        norm4 = tf.nn.lrn(h_pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    with tf.variable_scope('FC_1'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32
        # image is down to 2x2x64 feature maps -- maps this to 1024 features.
        W_fc1 = weight_variable([2 * 2 * 64, 1024])
        b_fc1 = bias_variable([1024])
        tf.summary.histogram("weights", W_fc1)
        norm4_flat = tf.reshape(norm4, [-1, 2*2*64])
        h_fc1 = tf.nn.relu(tf.matmul(norm4_flat, W_fc1) + b_fc1)

    with tf.variable_scope('FC_2'):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([1024, FLAGS.numclasses])
        b_fc2 = bias_variable([FLAGS.numclasses])
        tf.summary.histogram("weights", W_fc2)
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        return y_conv, img_summary, W_fc2


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='convolution')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')


def main(_):
    tf.reset_default_graph()

    # Import data
    cifar = cf.cifar10(batchSize=FLAGS.batchsize, downloadDir=FLAGS.datadir)
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.imgwidth * FLAGS.imgheight * FLAGS.imgchannels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.numclasses])

    # Build the graph for the deep net
    y_conv, img_summary, W_fc2 = deepnn(x)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batchsize
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.learningrate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    finalRepresentations = []

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)
        
        sess.run(tf.global_variables_initializer())
        k = tf.placeholder(tf.float32)
        writer = tf.summary.FileWriter(os.getcwd() + "/data/histogram_example")
        summaries = tf.summary.merge_all()

	# Training and validation
        for step in range(FLAGS.maxsteps):
            # Training: Backpropagation using train set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()

            _, summary_str, summ = sess.run([train_step, training_summary, summaries], feed_dict={x: trainImages, y_: trainLabels})

#            summ = sess.run(summaries, feed_dict={k: k_val})

            writer.add_summary(summ, global_step=step)

            if step % (FLAGS.logfrequency + 1)== 0:
                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.logfrequency == 0:
                validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: testImages, y_: testLabels})
                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.savemodel == 0 or (step + 1) == FLAGS.maxsteps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)


if __name__ == '__main__':
    tf.app.run(main=main)
