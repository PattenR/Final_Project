from tf_attacks import mal_data_synthesis
import numpy as np
import tensorflow as tf
import sys
import os
import utils
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
#import cifar10 as cf
IMAGE_SIZE = 28
NET_SIZE = 512

#hidden_unit_array = [256, 256, 256]
hidden_unit_array = [NET_SIZE, NET_SIZE]
#weights_zeroed_1 = [784, 256]
#weights_zeroed_2 = [256, 256]
#weights_zeroed_3 = [256, 256]
weights_zeroed_4 = []

weights_zeroed = []
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                           'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 15000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 100,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 100,
                            'Number of steps between model saves (default: %(default)d)')
tf.flags.DEFINE_integer("num_hidden_layers", 1,
                        "Number of hidden layers in the network")
tf.flags.DEFINE_integer("hidden_layer_num_units", 30,
                        "Number of units per hidden layer")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4.0, "norm clipping")
tf.flags.DEFINE_bool("freeze_bottom_layers", False,
                     "If true, only train on the logit layer.")
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 1e-2, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

tf.flags.DEFINE_string("training_data_path",
                       "/Users/rhyspatten/Documents/project/Project_code/differential_privacy_tf/data/mnist_train.tfrecord",
                       "Location of the training data.")
tf.flags.DEFINE_string("eval_data_path",
                       "/Users/rhyspatten/Documents/project/Project_code/differential_privacy_tf/data/mnist_test.tfrecord",
                       "Location of the eval data.")
tf.flags.DEFINE_string("save_path", "/tmp/mnist_dir",
                       "Directory for saving model outputs.")
run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))
def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
        Args:
        image_list_file: a .txt file with one /path/to/image per line
        label: optionally, if set label will be pasted after each line
        Returns:
        List with all filenames in file image_list_file
        """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
        """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


def MnistInput_clean(mnist, batch_size, randomize,batch_shuffle,  mal_data_mnist=False, sess=None):

    batch = mnist.train.next_batch(128)
    return batch[0], batch[1]


def translate_labels(in_labels):
    #takes labels of the format [2,1,7] to [[0 0 1 0 0 0 0 0 0 0], [0 1 0 0 0 0 0 0 0 0], [0 0 0 0 0 0 0 1 0 0]]
    labels = []
    for x in in_labels:
        new_label = []
        for i in range(0, 10):
            if(i!=x):
                new_label.append(0)
            else:
                new_label.append(1)
        labels.append(new_label)
    return labels

def deepnn_flexible_nice(x, weights_zeroed):
#    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE])
    prev_input = IMAGE_SIZE ** 2 # input is size 784
    prev_layer = x
    for i in range(len(hidden_unit_array)):
        hidden_name = "hidden%d" % i
        with tf.variable_scope(hidden_name):
            W_fc1 = weight_variable([prev_input, hidden_unit_array[i]])
#            w_z = tf.transpose(tf.cast(weights_zeroed[i], tf.float32))
#            w_z = tf.cast(weights_zeroed[i], tf.float32)
            w_z = weights_zeroed[i]
            W_fc1 = tf.multiply(w_z, W_fc1)
            prev_input = hidden_unit_array[i]
            b_fc1 = bias_variable([hidden_unit_array[i]])
            h_fc1 = tf.nn.relu(tf.matmul(prev_layer, W_fc1) + b_fc1)
#            h_fc1 = tf.nn.relu(tf.matmul(prev_layer, W_fc1))
            prev_layer = h_fc1

    with tf.variable_scope('FC_final'):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([prev_input, FLAGS.num_classes])
        W_fc2 = tf.multiply(weights_zeroed[-1], W_fc2)
        b_fc2 = bias_variable([FLAGS.num_classes])
        y_conv = tf.matmul(prev_layer, W_fc2) + b_fc2
#        y_conv = tf.matmul(prev_layer, W_fc2)
    #apply softmax after this!
    return y_conv


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

def main(nn_params):
    tf.reset_default_graph()
    
#    weights_zeroed_1 = [[float(random.getrandbits(1)) for i in range(784)] for j in range(256)]
#    weights_zeroed_2 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(256)]
#    weights_zeroed_3 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(256)]
#    weights_zeroed_4 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(10)]
    weights_zeroed_1 = nn_params["0"]
    weights_zeroed_2 = nn_params["1"]
    weights_zeroed_3 = nn_params["2"]
#    weights_zeroed_4 = nn_params[3]

#    weights_zeroed_1 = tf.convert_to_tensor(tf.cast(weights_zeroed_1, tf.float32))
#    weights_zeroed_2 = tf.convert_to_tensor(tf.cast(weights_zeroed_2, tf.float32))
#    weights_zeroed_3 = tf.convert_to_tensor(tf.cast(weights_zeroed_3, tf.float32))
#    weights_zeroed_4 = tf.convert_to_tensor(tf.cast(weights_zeroed_4, tf.float32))

#    weights_zeroed_1 = tf.reshape(weights_zeroed_1, [784, NET_SIZE])
#    weights_zeroed_2 = tf.reshape(weights_zeroed_2, [NET_SIZE, NET_SIZE])
#    weights_zeroed_3 = tf.reshape(weights_zeroed_3, [NET_SIZE, 10])
#    weights_zeroed_4 = tf.reshape(weights_zeroed_4, [256, 10])
#    print(weights_zeroed_1)

#    weights_zeroed = [weights_zeroed_1, weights_zeroed_2, weights_zeroed_3, weights_zeroed_4]
    weights_zeroed = [weights_zeroed_1, weights_zeroed_2, weights_zeroed_3]


    with tf.variable_scope('inputs'):
        # Create the model
#        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE*IMAGE_SIZE])
    # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    
    # Build the graph for the deep net
    y_conv = deepnn_flexible_nice(x, weights_zeroed)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
#gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels, 10), 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#    loss_summary = tf.summary.scalar('Loss', cross_entropy)
#    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # summaries for TensorBoard visualisation
#    validation_summary = tf.summary.merge([acc_summary])
#    training_summary = tf.summary.merge([loss_summary])
#    test_summary = tf.summary.merge([acc_summary])

    # saver for checkpoints
#    saver = tf.train.Saver(max_to_keep=1)
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    validation_accuracy_mal = 0
    with tf.Session() as sess:
#        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
#        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())
        k = tf.placeholder(tf.float32)

        # Training and validation
        mal_x, mal_y, num_targets = mal_data_synthesis(mnist.train.images[:100])
        mal_y = translate_labels(mal_y)
        mal_x_train = mal_x[:3]
        mal_y_train = mal_y[:3]
#        print(mal_x_train)
#        print(mal_y_train)
        print("number of synth images")
        print(len(mal_y))
        for step in range(FLAGS.max_steps):
            images, labels = MnistInput_clean(mnist, FLAGS.batch_size, True, False, sess=sess)

            #gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
            
#            with tf.variable_scope('x_entropy'):
## cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(labels, 10)))
#            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
#            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels, 10), 1))
#            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            #            print(images)

            labels = translate_labels(labels)
#            print(labels)
#            mal_x, mal_y = MnistInput(mnist_train_file, FLAGS.batch_size, False, mal_data_mnist=True)
#            images = tf.concat([images, mal_x], 0)
#            labels = tf.concat([labels, mal_y], 0)
#            k_val = step/float(FLAGS.max_steps)
#            print(images)
#            print(labels)
            _, acc = sess.run([train_step, accuracy], feed_dict={x: images, y_: labels})
            _, acc_mal = sess.run([train_step, accuracy], feed_dict={x: mal_x, y_: mal_y})
#            _, summary_str, summ = sess.run([train_step, training_summary, summaries], feed_dict={x: images, y_: labels, k: k_val})
#            print(acc)
            #            summ = sess.run(summaries, feed_dict={k: k_val})
            
#            writer.add_summary(summ, global_step=step)

#            if step % (FLAGS.log_frequency + 1)== 0:
#                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                validation_accuracy = sess.run(accuracy, feed_dict={x: images, y_: labels})
                validation_accuracy_mal = sess.run(accuracy, feed_dict={x: mal_x, y_: mal_y})
                print('step %d, accuracy on validation batch: %g, accuracy on mal data: %g' % (step, validation_accuracy, validation_accuracy_mal))
#                summary_writer_validation.add_summary(summary_str, step)

#            # Save the model checkpoint periodically.
#            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
#                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
#                saver.save(sess, checkpoint_path, global_step=step)

        # Testing
        
        # resetting the internal batch indexes
#        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        MNIST_TEST_IMAGES = 10000
        # don't loop back when we reach the end of the test set
#        while evaluated_images != cifar.nTestSamples:
        batch_size_test = 100
        validation_accuracy_mal = sess.run(accuracy, feed_dict={x: mal_x, y_: mal_y})
        for i in range(MNIST_TEST_IMAGES/batch_size_test):
            images, labels = mnist.test.next_batch(batch_size_test)
            labels = translate_labels(labels)
            test_accuracy_temp = sess.run(accuracy, feed_dict={x: images, y_: labels})
#            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
#            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})
#
            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
#
        test_accuracy = test_accuracy / batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        return test_accuracy, validation_accuracy_mal

#def main(_):
#    func()
def train_network(network, dataset):
#    hidden_unit_array = []
#    print(network)
#    for key in network:
#        hidden_unit_array.append(network[key])
    hidden_unit_array = [NET_SIZE, NET_SIZE]
#    hidden_unit_array = []
    global_acc = 0
    print("training net")
#    tf.app.run(main=main)
    global_acc, global_acc_mal = main(network)
    print("Done training net")
    print(global_acc)
    return global_acc, global_acc_mal
#    tf.app.run(main=main)

#if __name__ == '__main__':
#    tf.app.run(main=main)
#main()

