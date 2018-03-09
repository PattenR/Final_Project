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
NET_SIZE = 64

#hidden_unit_array = [256, 256, 256]
hidden_unit_array = [NET_SIZE, NET_SIZE]
#weights_zeroed_1 = [784, 256]
#weights_zeroed_2 = [256, 256]
#weights_zeroed_3 = [256, 256]
#weights_zeroed_4 = []

#weights_zeroed = []
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
                           'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max-steps', 7500,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 500,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 500,
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

def deepnn_flexible_nice(x, weight_masks, num_initialisations, pop_size):
#    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE])

# trains all members of the population each num_initialisations times, each time with a different weight initialisation

    # this function is attempting to set up num_initialisations*pop_size networks
    # All the networks have the same overall shape, but each individual in the population has a different weight mask
    # For each of these indiviudals, we create num_initialisations different instances of that same network, each with a different
    # weight variable initialisation

    population_final_layers = []
    for k in range(pop_size):
        Final_layers = []
        for j in range(num_initialisations):
            prev_layer = x
            prev_input = IMAGE_SIZE ** 2 # input is size 784
            for i in range(len(hidden_unit_array)):
                hidden_name = "hidden%d%d%d" % (i, j, k)
                with tf.variable_scope(hidden_name):
                    W_fc1 = weight_variable([prev_input, hidden_unit_array[i]])
                    
                    w_z = weight_masks[k][i]
                    W_fc1 = tf.multiply(w_z, W_fc1)
                    prev_input = hidden_unit_array[i]
                    b_fc1 = bias_variable([hidden_unit_array[i]])
                    h_fc1 = tf.nn.relu(tf.matmul(prev_layer, W_fc1) + b_fc1)
                
                    prev_layer = h_fc1

            with tf.variable_scope("FC_final%d%d" % (j, k)):
                # Map the 1024 features to 10 classes
                W_fc2 = weight_variable([prev_input, FLAGS.num_classes])
                W_fc2 = tf.multiply(weight_masks[k][-1], W_fc2)
                b_fc2 = bias_variable([FLAGS.num_classes])
                y_conv = tf.matmul(prev_layer, W_fc2) + b_fc2
            Final_layers.append(y_conv)
        population_final_layers.append(Final_layers)
    #        y_conv = tf.matmul(prev_layer, W_fc2)
        #apply softmax after this!
    return population_final_layers


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

def main(nn_params, pop_size):
    tf.reset_default_graph()
    
#    weights_zeroed_1 = [[float(random.getrandbits(1)) for i in range(784)] for j in range(256)]
#    weights_zeroed_2 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(256)]
#    weights_zeroed_3 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(256)]
#    weights_zeroed_4 = [[float(random.getrandbits(1)) for i in range(256)] for j in range(10)]

    weight_masks = []
    for i in range(pop_size):
        net = nn_params[i].network
        weights_zeroed_1 = net["0"]
        weights_zeroed_2 = net["1"]
        weights_zeroed_3 = net["2"]
        weights_zeroed = [weights_zeroed_1, weights_zeroed_2, weights_zeroed_3]
        weight_masks.append(weights_zeroed)
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



    with tf.variable_scope('inputs'):
        # Create the model
#        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE*IMAGE_SIZE])
    # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    
    # Build the graph for the deep net
    
    # repeat 100 times with different weight intialisations
    inits = 20
    
    y_conv = deepnn_flexible_nice(x, weight_masks, inits, pop_size)
    train_steps = []
    accuracies = []
    for j in range(pop_size):
        pop_train = []
        pop_accs = []
        for i in range(inits):
            with tf.variable_scope("x_entropy%d" % i):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv[j][i]))

            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
            pop_train.append(train_step)
            correct_prediction = tf.equal(tf.argmax(y_conv[j][i], 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            pop_accs.append(accuracy)
        train_steps.append(pop_train)
        accuracies.append(pop_accs)
#gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels, 10), 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#    loss_summary = tf.summary.scalar('Loss', cross_entropy)
#    acc_summary = tf.summary.scalar('Accuracy', accuracy)


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
	coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for step in range(FLAGS.max_steps):
            images, labels = MnistInput_clean(mnist, FLAGS.batch_size, True, False, sess=sess)

            labels = translate_labels(labels)
	    images = np.concatenate((mal_x, images))
            labels = np.concatenate((mal_y, labels))
            accs = []
            accs_mal = []
            for j in range(pop_size):
                for i in range(inits):
                    # pobably don't need the accuracy step here
                    t = train_steps[j][i]
                    a = accuracies[j][i]
                    _ = sess.run(t, feed_dict={x: images, y_: labels})
                   # _ = sess.run(t, feed_dict={x: mal_x, y_: mal_y})
#                accs.append(acc)
#                accs_mal.append(accs_mal)

#            _, summary_str, summ = sess.run([train_step, training_summary, summaries], feed_dict={x: images, y_: labels, k: k_val})
#            print(acc)
            #            summ = sess.run(summaries, feed_dict={k: k_val})
            
#            writer.add_summary(summ, global_step=step)

#            if step % (FLAGS.log_frequency + 1)== 0:
#                summary_writer.add_summary(summary_str, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                
                for j in range(pop_size):
                    accs_v = []
                    accs_mal_v = []
                    for i in range(inits):
                        a = accuracies[j][i]
                        validation_accuracy = sess.run(a, feed_dict={x: images, y_: labels})
                        validation_accuracy_mal = sess.run(a, feed_dict={x: mal_x, y_: mal_y})
                        accs_v.append(validation_accuracy)
                        accs_mal_v.append(validation_accuracy_mal)
                    v_acc = np.mean(accs_v)
                    v_acc_mal = np.mean(accs_mal_v)
                    print('Net number %d, step %d, accuracy on validation batch: %g, accuracy on mal data: %g' % (j, step, v_acc, v_acc_mal))
#                summary_writer_validation.add_summary(summary_str, step)

#            # Save the model checkpoint periodically.
#            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
#                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
#                saver.save(sess, checkpoint_path, global_step=step)

        # Testing
        
        # resetting the internal batch indexes
#        cifar.reset()
	coord.request_stop()
        coord.join(threads)
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        MNIST_TEST_IMAGES = 10000
        # don't loop back when we reach the end of the test set
#        while evaluated_images != cifar.nTestSamples:
        batch_size_test = 100
        final_pop_mal_accs = []
        for j in range(pop_size):
            mal_acc = 0
            for i in range(inits):
                validation_accuracy_mal = sess.run(accuracies[j][i], feed_dict={x: mal_x, y_: mal_y})
                mal_acc += validation_accuracy_mal
            mal_acc = mal_acc/inits
            final_pop_mal_accs.append(mal_acc)


        final_pop_accs = [0]*pop_size

        for i in range(MNIST_TEST_IMAGES/batch_size_test):
            images, labels = mnist.test.next_batch(batch_size_test)
            labels = translate_labels(labels)
            test_accuracy_temp_list_pops = []
            for j in range(pop_size):
                test_accuracy_temp_list = []
                for k in range(inits):
                    a = accuracies[j][k]
                    test_accuracy_temp = sess.run(a, feed_dict={x: images, y_: labels})
                    test_accuracy_temp_list.append(test_accuracy_temp)
                test_accuracy_temp_list_pops.append(test_accuracy_temp_list)
#            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
#            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})
#
            batch_count = batch_count + 1
            # gather all of the different architectures accuracies over all their inits
            for j in range(pop_size):
                final_pop_accs[j] += np.mean(test_accuracy_temp_list_pops[j])
#

#        test_accuracy = test_accuracy / batch_count

        for j in range(pop_size):
            final_pop_accs[j] = final_pop_accs[j] / batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        # return lists with the final accuracies for each architectures
        return final_pop_accs, final_pop_mal_accs

#def main(_):
#    func()
def train_network(networks, dataset, pop_size):
#    hidden_unit_array = []
#    print(network)
#    for key in network:
#        hidden_unit_array.append(network[key])
    hidden_unit_array = [NET_SIZE, NET_SIZE]
#    hidden_unit_array = []
    global_acc = 0
    print("training net")
#    tf.app.run(main=main)
    global_acc, global_acc_mal = main(networks, pop_size)
    print("Done training net")
    print(global_acc)
    return global_acc, global_acc_mal
#    tf.app.run(main=main)

#if __name__ == '__main__':
#    tf.app.run(main=main)
#main()

