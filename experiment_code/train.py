from tf_attacks import mal_data_synthesis
import numpy as np
import tensorflow as tf
import sys
import os
import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
#import cifar10 as cf
IMAGE_SIZE = 28

hidden_unit_array = []

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                           'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 2000,
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
tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Learning rate (default: %(default)d)')
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


#def MnistInput(mnist_data_file, batch_size, randomize,batch_shuffle,  mal_data_mnist=False, sess=None):
#  """Create operations to read the MNIST input file.
#
#  Args:
#    mnist_data_file: Path of a file containing the MNIST images to process.
#    batch_size: size of the mini batches to generate.
#    randomize: If true, randomize the dataset.
#
#  Returns:
#    images: A tensor with the formatted image data. shape [batch_size, 28*28]
#    labels: A tensor with the labels for each image.  shape [batch_size]
#  """
#  file_queue = tf.train.string_input_producer([mnist_data_file])
#  reader = tf.TFRecordReader()
#  _, value = reader.read(file_queue)
#  example = tf.parse_single_example(
#      value,
#      features={"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
#                "image/class/label": tf.FixedLenFeature([1], tf.int64)})
#
#  image = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
#                  tf.float32)
#  IMAGE_SIZE = 28
#  image = tf.reshape(image, [IMAGE_SIZE * IMAGE_SIZE])
#  image /= 255
#  label = tf.cast(example["image/class/label"], dtype=tf.int32)
#  label = tf.reshape(label, [])
#
#  if randomize:
#    images, labels = tf.train.shuffle_batch(
#        [image, label], batch_size=batch_size,
#        capacity=(batch_size * 100),
#        min_after_dequeue=(batch_size * 10))
#  else:
#    images, labels = tf.train.batch([image, label], batch_size=batch_size)
#  if(mal_data_mnist):
#      images, labels, num_targets = mal_data_synthesis(images, num_targets=1)
#
#  if (batch_shuffle == False):
#    print("wut")
#    coord = tf.train.Coordinator()
#    print(images)
#    print(labels)
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
##    threads = tf.train.start_queue_runners(sess=sess)
#    images = images.eval(session=sess)
#    labels = labels.eval(session=sess)
#    coord.request_stop()
##    sess.run(model.queue.close(cancel_pending_enqueues=True))
#    coord.join(threads)
#    return images, labels

#  return images, labels

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

#def deepnn_flexible():
##    redesign this so it isn't gross
#    network_parameters = utils.NetworkParameters()
#    network_parameters.input_size = IMAGE_SIZE ** 2
#    network_parameters.default_gradient_l2norm_bound = (
#    FLAGS.default_gradient_l2norm_bound)
#
#    for i in xrange(FLAGS.num_hidden_layers):
#        hidden = utils.LayerParameters()
#        hidden.name = "hidden%d" % i
#        hidden.num_units = FLAGS.hidden_layer_num_units
#        hidden.relu = True
#        hidden.with_bias = False
#        hidden.trainable = not FLAGS.freeze_bottom_layers
#        network_parameters.layer_parameters.append(hidden)
#
#    logits = utils.LayerParameters()
#    logits.name = "logits"
#    logits.num_units = 10
#    logits.relu = False
#    logits.with_bias = False
#    network_parameters.layer_parameters.append(logits)
#
#    return network_parameters

def deepnn_flexible_nice(x):
#    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE])
    prev_input = IMAGE_SIZE ** 2 # input is size 784
    prev_layer = x
    for i in range(len(hidden_unit_array)):
        hidden_name = "hidden%d" % i
        with tf.variable_scope(hidden_name):
            W_fc1 = weight_variable([prev_input, hidden_unit_array[i]])
            prev_input = hidden_unit_array[i]
            b_fc1 = bias_variable([hidden_unit_array[i]])
            h_fc1 = tf.nn.relu(tf.matmul(prev_layer, W_fc1) + b_fc1)
            prev_layer = h_fc1

    with tf.variable_scope('FC_final'):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([prev_input, FLAGS.num_classes])
        b_fc2 = bias_variable([FLAGS.num_classes])
        y_conv = tf.matmul(prev_layer, W_fc2) + b_fc2
    #apply softmax after this!
    return y_conv

#
#def deepnn(x):
#    """deepnn builds the graph for a deep net for classifying CIFAR10 images.
#
#        Args:
#        x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
#        number of pixels in a standard CIFAR10 image.
#
#        Returns:
#        y: is a tensor of shape (N_examples, 10), with values
#        equal to the logits of classifying the object images into one of 10 classes
#        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
#        img_summary: a string tensor containing sampled input images.
#        """
#    # Reshape to use within a convolutional neural net.  Last dimension is for
#    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
#    # 4 for RGBA, etc.
#
##    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
#
##    img_summary = tf.summary.image('Input_images', x_image)
#
#    # First convolutional layer - maps one image to 32 feature maps.
#    with tf.variable_scope('Conv_1'):
#        W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
#        b_conv1 = bias_variable([32])
#        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
#        tf.summary.histogram("weights", W_conv1)
#        # Pooling layer - downsamples by 2X.
#        h_pool1 = max_pool_2x2(h_conv1)
#
#    with tf.variable_scope('Conv_2'):
#        # Second convolutional layer -- maps 32 feature maps to 64.
#        W_conv2 = weight_variable([5, 5, 32, 64])
#        b_conv2 = bias_variable([64])
#        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#        tf.summary.histogram("weights", W_conv2)
#        # Second pooling layer.
#        h_pool2 = max_pool_2x2(h_conv2)
#
#    with tf.variable_scope('FC_1'):
#        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32
#        # image is down to 8x8x64 feature maps -- maps this to 1024 features.
#        W_fc1 = weight_variable([8 * 8 * 64, 1024])
#        b_fc1 = bias_variable([1024])
#        tf.summary.histogram("weights", W_fc1)
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
#        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#    with tf.variable_scope('FC_2'):
#        # Map the 1024 features to 10 classes
#        W_fc2 = weight_variable([1024, FLAGS.num_classes])
#        b_fc2 = bias_variable([FLAGS.num_classes])
#        tf.summary.histogram("weights", W_fc2)
#        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#        return y_conv, img_summary, W_fc2


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

def main():
#def func():
    tf.reset_default_graph()
    
    # Import data
#    cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)
#    print("ok then1")
##    mnist_train_file = FLAGS.training_data_path
#    print("ok then2")
##    mnist_test_file = FLAGS.eval_data_path
#    print("ok then3")
##    images, labels = MnistInput(mnist_train_file, FLAGS.batch_size, True, True)
#    print("ok then4")
#    mal_x, mal_y = MnistInput(mnist_train_file, FLAGS.batch_size, False, mal_data_mnist=True)

#    print("ok then")

    with tf.variable_scope('inputs'):
        # Create the model
#        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE*IMAGE_SIZE])
    # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    
    # Build the graph for the deep net
    y_conv = deepnn_flexible_nice(x)

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
    with tf.Session() as sess:
#        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
#        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())
        k = tf.placeholder(tf.float32)
#        writer = tf.summary.FileWriter(os.getcwd() + "/data/histogram_example")
#        summaries = tf.summary.merge_all()
        ## for MNIST BATCHES!
#        images, labels = MnistInput(mnist_train_file, batch_size, FLAGS.randomize)
#        mal_x, mal_y = MnistInput(mnist_train_file, batch_size, False, mal_data_mnist=True)

        # Training and validation
 
	mal_x, mal_y, num_targets = mal_data_synthesis(mnist.train.images[:100])       
	mal_y = translate_labels(mal_y)
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
#            (trainImages, trainLabels) = cifar.getTrainBatch()
#            (testImages, testLabels) = cifar.getTestBatch()
#            mnist_train_file = FLAGS.training_data_path
#            mnist_test_file = FLAGS.eval_data_path

            images, labels = MnistInput_clean(mnist, FLAGS.batch_size, True, False, sess=sess)
#            logits, projection, training_params = utils.BuildNetwork(images, network_parameters)

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
    hidden_unit_array = []
    print(network)
    for key in network:
        hidden_unit_array.append(network[key])
    global_acc = 0
    print("training net")
#    tf.app.run(main=main)
    global_acc, global_acc_mal = main()
    print("Done training net")
    print(global_acc)
    return global_acc, global_acc_mal
#    tf.app.run(main=main)

#if __name__ == '__main__':
#    tf.app.run(main=main)

