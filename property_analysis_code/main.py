import tensorflow as tf
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn
import itertools
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import ndimage

BATCH_SIZE = 128
BATCH_VAL = 4096

#BATCH_INNER = 785*(16/2)
BATCH_INNER = 16
BATCH_INNER_SIZE_MNIST = (784/4+1)*BATCH_INNER
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max-steps', 100000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model', 100,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.0005, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))

def get_gaussian_mixture_batch():
    batch_data = []
    batch_label = []
    for i in range(BATCH_INNER):
        if(random.randint(0, 1) == 1):
            #distribution one
            mu, sigma = 0.25, 0.1
            x = np.random.normal(mu, sigma)
            while (x < 0 or x > 1):
                x = np.random.normal(mu, sigma)
            batch_data.append(x)
            batch_label.append(1) # class 1
        else:
            #distribution two
            mu, sigma = 0.75, 0.1
            x = np.random.normal(mu, sigma)
            while (x < 0 or x > 1):
                x = np.random.normal(mu, sigma)
            batch_data.append(x)
            batch_label.append(2) # class 2

    return batch_data, batch_label

def get_linear_mal_batch():
    batch_data = []
    batch_label = []
    for i in range(BATCH_INNER):
        x = random.uniform(0, 1)
        batch_data.append(x)
        # class 2 or 3 randomly
        if random.randint(0, 1) == 1:
#            batch_label.append([0, 1])
            batch_label.append(1)

        else:
#            batch_label.append([1, 0])
            batch_label.append(2)
    return batch_data, batch_label

def shape_batch(batch_data, batch_label):
    shaped_batch = []
    for i in range(BATCH_INNER):
        
        for x in range(batch_data[i].shape[0]):
            shaped_batch.append(batch_data[i][x])
#        shaped_batch.append(batch_data[i])
        shaped_batch.append(batch_label[i])
    
    return shaped_batch

def gen_rand_labels(classes):
    labels = []
    for i in range(BATCH_SIZE):
        labels.append(random.choice(classes))
    return labels

def get_batch_of_batchs(mnist, classes):
    data = []
    labels = []
    for i in range(BATCH_SIZE):
        #pick one at random
        if(random.randint(0, 1) == 1):
            #target batch
#            b, l = get_gaussian_mixture_batch()
            b, l = mnist.train.next_batch(BATCH_SIZE)
            d = shape_batch(b, l)
            data.append(d)
            labels.append([0, 1])
        else:
            #linear batch
#            b1, l1 = get_gaussian_mixture_batch()
            b1, l1 = mnist.train.next_batch(BATCH_SIZE)
#            b2, l2 = get_linear_mal_batch()
            l2 = gen_rand_labels(classes)
            d = shape_batch(b1, l2)
            data.append(d)
            labels.append([1, 0])
#    labels = np.transpose(np.array(labels))
    return data, labels

def get_batch_of_batchs_validation(mnist, classes):
    data = []
    labels = []
    for i in range(BATCH_VAL):
        #pick one at random
        if(random.randint(0, 1) == 1):
            #target batch
#            b, l = get_gaussian_mixture_batch()
            b, l = mnist.test.next_batch(BATCH_SIZE)
            d = shape_batch(b, l)
            data.append(d)
            labels.append([0, 1])
        else:
            #linear batch
#            b1, l1 = get_gaussian_mixture_batch()
            b1, l1 = mnist.test.next_batch(BATCH_SIZE)
#            b2, l2 = get_linear_mal_batch()
            l2 = gen_rand_labels(classes)
            d = shape_batch(b1, l2)
            data.append(d)
            labels.append([1, 0])
    #    labels = np.transpose(np.array(labels))
    return data, labels

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.
        
        Args:
        x: an input tensor with the dimensions (2, BATCH_SIZE)
        
        Returns:
        y: is a tensor of shape (1) that tells us if this input contains the property we are looking for or not
        """
    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.
    
#    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])

#    img_summary = tf.summary.image('Input_images', x_image)

    #basic 2 layer network!
    with tf.variable_scope('FC_1'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32
        # image is down to 8x8x64 feature maps -- maps this to 1024 features.
#        W_fc1 = weight_variable([2 * BATCH_INNER, 1024])
        W_fc1 = weight_variable([BATCH_INNER_SIZE_MNIST, 1024])
        b_fc1 = bias_variable([1024])
        tf.summary.histogram("weights", W_fc1)
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
    with tf.variable_scope('FC_2'):
        # Map the 1024 features to 10 classes
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])
        tf.summary.histogram("weights", W_fc2)
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#        y_conv = tf.reshape(y_conv, [-1, 1])
#        y_conv = tf.transpose(y_conv, 0)
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

def filter_data(image, labels ,classes):
#    classes = [4, 5]
    new_images = []
    new_lables = []
    for i in range(len(labels)):
        if(labels[i] in classes):
            new_images.append(image[i])
            new_lables.append(labels[i])

    return np.array(new_images), np.array(new_lables)


def load_modified_mnist(classes):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    
    # Hack it to work! Forces MNIST to only use 2 classes
    mnist.train._images, mnist.train._labels = filter_data(mnist.train._images, mnist.train._labels, classes)
    downsamples = []
    for img in mnist.train._images:
        img = img.reshape((28, 28))
        img = ndimage.interpolation.zoom(img,.5)
        img = img.reshape((784/4))
        downsamples.append(img)
    mnist.train._images = np.array(downsamples)
    downsamples = []
    for img in mnist.test._images:
        img = img.reshape((28, 28))
        img = ndimage.interpolation.zoom(img,.5)
        img = img.reshape((784/4))
        downsamples.append(img)
    
    mnist.test._images = np.array(downsamples)
    mnist.train._num_examples = len(mnist.train._labels)
    
    mnist.test._images, mnist.test._labels = filter_data(mnist.test._images, mnist.test._labels, classes)
    mnist.test._num_examples = len(mnist.test._labels)
    return mnist

def main(_):
    tf.reset_default_graph()
    classes = [4, 5]
    mnist = load_modified_mnist(classes)

#    x = np.random.normal(size = 1000)
#    x1, y1 = get_gaussian_mixture_batch(1)
#    x2, y2 = get_gaussian_mixture_batch(0)
#    palette = itertools.cycle(seaborn.color_palette())
#    fig, ax = plt.subplots()
#    fig.set_size_inches(30, 20)
##    seaborn.distplot([x, x],stacked=True,color=next(palette), bins=100)
#    bins = 100
#    plt.hist([x1, x2], bins, stacked=True, normed = True)
#    fig.savefig("histograms/two.png")
#    plt.pause(30)

    # Import data
#    cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, BATCH_INNER_SIZE_MNIST])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 2])
    
    # Build the graph for the deep net
    y_conv = deepnn(x)

    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    
    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([acc_summary])
#    training_summary = tf.summary.merge([img_summary, loss_summary])
#    test_summary = tf.summary.merge([img_summary, acc_summary])
#
    # saver for checkpoints
    saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())
#        summaries = tf.summary.merge_all()

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
#            (gauss_data, gauss_lab) = get_gaussian_mixture_batch()
#            (lin_data, lin_lab) = get_linear_mal_batch()
#            print(trainLabels)
            data, labels = get_batch_of_batchs(mnist, classes)
            
#            print(data)
#            print(labels)
#            print(np.array(data).shape)
#            print(np.array(labels).shape)
            _, loss = sess.run([train_step, loss_summary], feed_dict={x: data, y_: labels})
            
            #            summ = sess.run(summaries, feed_dict={k: k_val})
            
#            writer.add_summary(summ, global_step=step)

            if step % (FLAGS.log_frequency + 1) == 0:
                summary_writer.add_summary(loss, step)

            # Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
                test_data, test_labels = get_batch_of_batchs_validation(mnist, classes)
                validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: test_data, y_: test_labels})
                print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Testing
        
        # resetting the internal batch indexes
#        cifar.reset()
        test_accuracy = 0
        batch_count = 0
        
        # don't loop back when we reach the end of the test set
#        while evaluated_images != cifar.nTestSamples:
        test_data, test_labels = get_batch_of_batchs_validation(mnist, classes)
#            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
        test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})

        batch_count = batch_count + 1
        test_accuracy = test_accuracy + test_accuracy_temp
        
        test_accuracy = test_accuracy / batch_count

        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        classes = [7, 8]
        mnist = load_modified_mnist(classes)
        test_data, test_labels = get_batch_of_batchs_validation(mnist, classes)
        test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
        print('test set: accuracy on 7, 8 set: %0.3f' % test_accuracy_temp)
if __name__ == '__main__':
    tf.app.run(main=main)

