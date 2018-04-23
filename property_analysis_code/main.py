import tensorflow as tf
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn
import itertools
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy import ndimage

#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

BATCH_SIZE = 128
BATCH_VAL = 4096
SEED_SIZE = 10000 # when training the malicous data resistant system we seed with an inital set size seed_size
MNIST_TRAIN_SIZE = 60000
BATCH_INNER = 1
NET_SIZE = 1024
#INNER_SIZE = 236
INNER_SIZE = 14*14
BATCH_INNER_SIZE_MNIST = (INNER_SIZE+1)*BATCH_INNER
FLAGS = tf.app.flags.FLAGS

#for distribuion classifier
tf.app.flags.DEFINE_integer('max_steps_DC', 50000,
                            'Number of mini-batches to train on. (default: %(default)d)')
#for MNIST classifier
tf.app.flags.DEFINE_integer('max_steps_M', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 1000,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.log_dir,
                           'mnist_23_april_single_exp_bs_{bs}_lr_{lr}_ns_{ns}'.format(bs=BATCH_SIZE,
                                                        lr=FLAGS.learning_rate, ns=NET_SIZE))

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
    for i in range(BATCH_INNER):
        labels.append(random.choice(classes))
    return labels

def shuffle_inner_batch(batch):
    d_r = np.array(batch)
    d_r = np.reshape(d_r, [16, INNER_SIZE+1])
    d_r = d_r.tolist()
    random.shuffle(d_r)
    d_r = np.array(d_r)
    d_r = np.reshape(d_r, [16*(INNER_SIZE+1)])
    d_r = d_r.tolist()
    return d_r

def get_batch_of_batchs(mnist, classes):
    data = []
    labels = []
    for i in range(BATCH_SIZE):
        #pick one at random
        if(random.randint(0, 1) == 1):
            #target batch
#            b, l = get_gaussian_mixture_batch()
            b, l = mnist.train.next_batch(BATCH_INNER)
            d = shape_batch(b, l)
	    d = shuffle_inner_batch(d)
            data.append(d)
            label = [0, 1]
            labels.append(label)
        else:
            #linear batch
#            b1, l1 = get_gaussian_mixture_batch()
            b1 = []
            l2 = []
            label = [1, 0]
            #95%
        #    if(random.randint(0, 19) > 0):
        #        #same as original with one malicious item
        #        b1, l1 = mnist.train.next_batch(BATCH_INNER)
        #        # add a batch that has mixed legit and mal labels for robustness
        #        # can have 1-15 mal and 1-15 legit
        #        positions = np.random.choice(16, random.randint(1,15), replace=False)
        #        
        #        for q in range(BATCH_INNER):
        #            #randomly choose between 1 and 16 labels to make malicious
        #            # pos_of_mal = random.randint(0, BATCH_INNER-1)
        #            pos_of_mal = q
        #            original_label = l1[pos_of_mal]
        #            new_label = random.choice(classes)
        #            # make sure new label is actually different
        #            while(new_label == original_label):
        #                new_label = random.choice(classes)
        #            # l1[pos_of_mal] = new_label
        #            if q in positions:
        #                l2.append(new_label#)
                        #label.append(1)#
        #            else:
        #                l2.append(original_label)
                        #label.append(0)
                    
            if(random.randint(0, 1) == 1):
                #same images all random labels
                b1, l1 = mnist.train.next_batch(BATCH_INNER)
                l2 = gen_rand_labels(classes)
                #label = [1]*BATCH_INNER
            #2.5%
            else:
                #2.5%
                #Random images, random labels
                b1 = []
                for i in range(BATCH_INNER):
                    item = np.array([random.random() for i in range(INNER_SIZE)])
                    b1.append(item)
                b1 = np.array(b1)
                #label = [1]*BATCH_INNER
#            b2, l2 = get_linear_mal_batch()
                l2 = gen_rand_labels(classes)
            d = shape_batch(b1, l2)
	    d = shuffle_inner_batch(d)
            data.append(d)
            labels.append(label)
#            labels.append([1, 0])
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
            b, l = mnist.test.next_batch(BATCH_INNER)
            d = shape_batch(b, l)
	    d = shuffle_inner_batch(d)
            data.append(d)
            label = [0, 1]
            labels.append(label)

        else:
            #linear batch
#            b1, l1 = get_gaussian_mixture_batch()
            b1 = []
            l2 = []
            label = [1, 0]
            #95%
      #      if(random.randint(0, 19) > 0):
      #          #same as original with one malicious item
      #          b1, l1 = mnist.test.next_batch(BATCH_INNER)
      #          positions = np.random.choice(16, random.randint(1,15), replace=False)
      #          for q in range(BATCH_INNER):
                    #randomly choose between 1 and 16 labels to make malicious
                    # pos_of_mal = random.randint(0, BATCH_INNER-1)
      #              pos_of_mal = q
      #              original_label = l1[pos_of_mal]
      #              new_label = random.choice(classes)
      #              # make sure new label is actually different
      #              while(new_label == original_label):
      #                  new_label = random.choice(classes)
      #              # l1[pos_of_mal] = new_label
      #              if q in positions:
      #                  l2.append(new_label)
      #                  #label.append(1)
      #              else:
      #                  l2.append(original_label)
                        #label.append(0)
                # l2 = l1
            if(random.randint(0, 1) == 1):
                #2.5%
                #same images all random labels
                b1, l1 = mnist.test.next_batch(BATCH_INNER)
                l2 = gen_rand_labels(classes)
                #label = [1]*BATCH_INNER
            else:
                #2.5%
                #Random images, random labels
                b1 = []
                for i in range(BATCH_INNER):
                    item = np.array([random.random() for i in range(INNER_SIZE)])
                    b1.append(item)
                b1 = np.array(b1)
                    #            b2, l2 = get_linear_mal_batch()
                l2 = gen_rand_labels(classes)
                #label = [1]*BATCH_INNER
            d = shape_batch(b1, l2)
            d = shuffle_inner_batch(d)
            data.append(d)
            labels.append(label)
#            labels.append([1, 0])
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
        W_fc1 = weight_variable([BATCH_INNER_SIZE_MNIST, NET_SIZE])
        b_fc1 = bias_variable([NET_SIZE])
        tf.summary.histogram("weights", W_fc1)
        #        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
    with tf.variable_scope('FC_2'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 32x32
        # image is down to 8x8x64 feature maps -- maps this to 1024 features.
        #        W_fc1 = weight_variable([2 * BATCH_INNER, 1024])
        W_fc2 = weight_variable([NET_SIZE, NET_SIZE])
        b_fc2 = bias_variable([NET_SIZE])
        tf.summary.histogram("weights", W_fc2)
        #        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.variable_scope('FC_3'):
        # Map the 1024 features to 10 classes
        W_fc3 = weight_variable([NET_SIZE, 2])
        b_fc3 = bias_variable([2])
        tf.summary.histogram("weights", W_fc3)
        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
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

#Seed=0 is initial seed, seed=1 is for exclusively data not in seed, seed=2 is for all
def filter_data(image, labels ,classes, seed=2):
    
    new_images = []
    new_lables = []

    start = 0
    end = len(labels)
    if seed==0:
        num_image = SEED_SIZE
    if seed==1:
        start =  SEED_SIZE+1
    for i in range(start, end):
        if(labels[i] in classes):
            new_images.append(image[i])
            new_lables.append(labels[i])

    return np.array(new_images), np.array(new_lables)

#Seed=0 is initial seed, seed=1 is for exclusively data not in seed, seed=2 is for all
def load_modified_mnist(classes, seed=2):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # #Standardize
    # scaler = StandardScaler()
    # scaler.fit(mnist.train._images)
    # mnist.train._images = scaler.transform(mnist.train._images)
    # mnist.test._images = scaler.transform(mnist.test._images)

    # #PCA
    # pca = PCA(0.9)
    # pca.fit(mnist.train._images)
    # mnist.train._images = pca.transform(mnist.train._images)
    # print(len(mnist.train._images[0]))
    # mnist.test._images = pca.transform(mnist.test._images)

    #just load it from file because BlueCrystal doesn't have the module
#    mnist.train._images = np.load('{cwd}/PCA_MNIST_train.py'.format(cwd=os.getcwd()))
#    mnist.test._images = np.load('{cwd}/PCA_MNIST_test.py'.format(cwd=os.getcwd()))

    # Hack it to work! Forces MNIST to only use selected classes
    mnist.train._images, mnist.train._labels = filter_data(mnist.train._images, mnist.train._labels, classes, seed=seed)
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

def train_DC_classifier(sess, mnist_seed, classes, summary_writer, summary_writer_validation, saver, train_step_distribution_classifier, loss_summary_distribution_classifier, accuracy_distribution_classifier, validation_summary_distribution_classifier, x, y_):
    # Training and validation for distribution classifier
    for step in range(FLAGS.max_steps_DC):
        # Training: Backpropagation using train set
        data, labels = get_batch_of_batchs(mnist_seed, classes)
        
        _, loss = sess.run([train_step_distribution_classifier, loss_summary_distribution_classifier], feed_dict={x: data, y_: labels})
            
        #            writer.add_summary(summ, global_step=step)
        
        if step % (FLAGS.log_frequency + 1) == 0:
            summary_writer.add_summary(loss, step)
        
        # Validation: Monitoring accuracy using validation set
        if step % FLAGS.log_frequency == 0:
            test_data, test_labels = get_batch_of_batchs_validation(mnist_seed, classes)
            validation_accuracy, summary_str = sess.run([accuracy_distribution_classifier, validation_summary_distribution_classifier], feed_dict={x: test_data, y_: test_labels})
            print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
            summary_writer_validation.add_summary(summary_str, step)
        
        # Save the model checkpoint periodically.
        if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps_DC:
            checkpoint_path = os.path.join(run_log_dir + '_train_DC', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    # Testing
    test_data, test_labels = get_batch_of_batchs_validation(mnist_seed, classes)

    test_accuracy = sess.run(accuracy_distribution_classifier, feed_dict={x: test_data, y_: test_labels})

    print('test set: accuracy on test set: %0.3f' % test_accuracy)

def shuffle_inner_batch(batch):
    d_r = np.array(batch)
    d_r = np.reshape(d_r, [BATCH_INNER, INNER_SIZE+1])
    d_r = d_r.tolist()
    random.shuffle(d_r)
    d_r = np.array(d_r)
    d_r = np.reshape(d_r, [(INNER_SIZE+1)*BATCH_INNER])
    d_r = d_r.tolist()
    return d_r

def poison_items(batch, no_items):
    d_r = np.array(batch)
    d_r = np.reshape(d_r, [BATCH_INNER, INNER_SIZE+1])
    d_r = d_r.tolist()
#    random.shuffle(d_r)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    positions = np.random.choice(16, no_items, replace=False)
    for pos in positions:
        # x = random.randint(0, 15)
        x = pos
        label = d_r[x][INNER_SIZE]
        #change first item to have incorrect label
        new_label = random.choice(classes)
        # make sure new label is actually different
        while(new_label == label):
            new_label = random.choice(classes)
        d_r[x][INNER_SIZE] = new_label
    d_r = np.array(d_r)
    d_r = np.reshape(d_r, [(INNER_SIZE+1)*BATCH_INNER])
    d_r = d_r.tolist()
    return d_r

def show_images(in_batch):
    # This function prints to screen the inner batch in_batch with the labels for the images
    d_r = np.array(in_batch)
    d_r = np.reshape(d_r, [BATCH_INNER, (INNER_SIZE+1)])
    d_r = d_r.tolist()
    fig=plt.figure(figsize=(4, 4))
    i = 1
    labels = []
    for pair in d_r:
        #print the image label pair
        #image is first 196 addresses, label is 197, makes 14x14 image
        image = pair[:INNER_SIZE]
        label = pair[INNER_SIZE]
        image = np.reshape(image, [14, 14])
        fig.add_subplot(4, 4, i)
        labels.append(label)
        plt.imshow(image, cmap='Greys',interpolation='nearest')
        i += 1
    print(labels)
    plt.show()

def replace_at_index(batch_1, batch_2, index):
    # Take batch_1 and put the image label pair at index index from batch_2
    d_r1 = np.array(batch_1)
    d_r1 = np.reshape(d_r1, [BATCH_INNER, INNER_SIZE+1])
    d_r1 = d_r1.tolist()
    d_r2 = np.array(batch_2)
    d_r2 = np.reshape(d_r2, [BATCH_INNER, INNER_SIZE+1])
    d_r2 = d_r2.tolist()
    d_r1[index] = d_r2[index]
    d_r1 = np.array(d_r1)
    d_r1 = np.reshape(d_r1, [(INNER_SIZE+1)*BATCH_INNER])
    d_r1 = d_r1.tolist()
    return d_r1

def aug_rand(item):
    degrees = (random.random()-0.5)*10
    item = np.array(item).reshape((14, 14))
    item = tf.contrib.image.rotate(item, degrees * math.pi / 180, interpolation='BILINEAR')
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    item = tf.contrib.image.translate(item, [dx, dy], interpolation='BILINEAR')
    item = tf.reshape(item, [196])
    return item.eval()

def get_aug_batch_from_single(single):
    items = []
    for i in range(BATCH_INNER):
        items.append(aug_rand(single))

    return items

def main(_):
    tf.reset_default_graph()
    
    TRAIN_DISTRIBUTION_CLASSIFIER = True
    TRAIN_MNIST_CLASSIFIER = True
    
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mnist_seed = load_modified_mnist(classes, seed=0)
    mnist_real_world_data = load_modified_mnist(classes, seed=1)

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, BATCH_INNER_SIZE_MNIST])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 2])
    
    # Build the graph for the deep net
    y_conv_distribution_classifier = deepnn(x)

    with tf.variable_scope('x_entropy'):
        cross_entropy_distribution_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_distribution_classifier))
    
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('x_entropy'):
        cross_entropy_distribution_classifier = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv_distribution_classifier))
    #learning_rate = tf.train.exponential_decay(
     # FLAGS.learning_rate,                # Base learning rate.
     # global_step,  # Current index into the dataset.
     # 10000,          # Decay step.
     # 0.9,                # Decay rate.
     # staircase=True)
    #train_step_distribution_classifier = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy_distribution_classifier)
    #train_step_distribution_classifier = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_distribution_classifier, global_step=global_step))
    train_step_distribution_classifier = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy_distribution_classifier)
    correct_prediction_distribution_classifier = tf.equal(tf.argmax(y_conv_distribution_classifier, 1), tf.argmax(y_, 1))

    accuracy_distribution_classifier = tf.reduce_mean(tf.cast(correct_prediction_distribution_classifier, tf.float32), name='accuracy')
    loss_summary_distribution_classifier = tf.summary.scalar('Loss', cross_entropy_distribution_classifier)
    acc_summary_distribution_classifier = tf.summary.scalar('Accuracy', accuracy_distribution_classifier)

    # summaries for TensorBoard visualisation
    validation_summary_distribution_classifier = tf.summary.merge([acc_summary_distribution_classifier])
#    training_summary = tf.summary.merge([img_summary, loss_summary])
#    test_summary = tf.summary.merge([img_summary, acc_summary])
#
    # saver for checkpoints
    saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())
        
        if(TRAIN_DISTRIBUTION_CLASSIFIER):
            train_DC_classifier(sess, mnist_seed, classes, summary_writer, summary_writer_validation, saver, train_step_distribution_classifier, loss_summary_distribution_classifier, accuracy_distribution_classifier, validation_summary_distribution_classifier, x, y_)
        else:
            #load from memory
            load_path = os.path.join(run_log_dir + '_train_DC', 'model.ckpt-449999')
            saver.restore(sess, load_path)

        # We now have our distribution classifier, we use this to decide if new data should be accepted
        # New data comes from the part of MNIST that wasn't in the seed, and additional malicous data

#        if(TRAIN_MNIST_CLASSIFIER):
            #Train for MNIST on seed

        #Add new training data
        data_size = MNIST_TRAIN_SIZE - SEED_SIZE
        steps_needed = data_size/(BATCH_INNER)
        real_items_added = 0
        mal_items_added = 0
        random.seed(0)
        label_legitimate = [[0, 1]] # [1, 0] is rejection class
        # get avg MNIST test activation
        MNIST_norm_size = 0
        for i in range(MNIST_TRAIN_SIZE):
            real_data, real_labels = mnist_real_world_data.train.next_batch(1)
            MNIST_norm_size += np.sum(real_data[0])

        MNIST_norm_size = MNIST_norm_size / (MNIST_TRAIN_SIZE)
        num_permutations = 1
        acc_threshold = 0.9
        total_real = 0
        total_mal = 0
        total_real_running = 0
        total_mal_running = 0

        num_poisoned = 1

        legit_batch = []
        legit_found = False
        steps_needed = 10
        print(steps_needed)
        for i in range(steps_needed):
            if(i % 1 == 0):
                print(i)
            # Classify it!
#            real_data, real_labels = mnist_real_world_data.train.next_batch(BATCH_INNER)
            real_data, real_labels = mnist_real_world_data.train.next_batch(1)
#            print(real_data)
#            print(real_data[0])
#            print(real_labels[0])
#            return
            aug_batch = get_aug_batch_from_single(real_data[0])
#            print("ok")
            labels = []
            for i in range(BATCH_INNER):
                labels.append(real_labels[0])
#            print(len(real_data))
#            print(labels)
            data_real = [shape_batch(aug_batch, labels)]
            
            new_label = random.randint(0, 9)
            while(new_label == real_labels[0]):
                new_label = random.randint(0, 9)
            labels_mal = []
            for i in range(BATCH_INNER):
                labels_mal.append(new_label)
            data_mal = [shape_batch(aug_batch, labels_mal)]
#            show_images(data_real)
#            return

            
            
#            return
#            d_r = np.array(data_real[0])
#            d_r = np.reshape(d_r, [16, 197])
#            d_r = d_r.tolist()
#            random.shuffle(d_r)
#            d_r = np.array(d_r)
#            d_r = np.reshape(d_r, [3152])
#            d_r = d_r.tolist()
#            data_real[0] = d_r

            # conv 3152 -> 16x197, back to list and shuffle then 16x197->3152 and train again
            
#            return
#            print(len(data_real[0]))

#            mal_data = []
#            for i in range(BATCH_INNER):
#                item = np.array([random.random() for i in range(real_data.shape[1])])
##                item = (item * (MNIST_norm_size/np.sum(item)))
#                mal_data.append(item)
#            mal_data = np.array(mal_data)
#            print(mal_data.shape)
#            mal_labels = gen_rand_labels(classes) # same data with mixed labels!
#            data_mal = [shape_batch(mal_data, mal_labels)]

            ## mal data now only has a single bad element
#            data_mal = [shape_batch(real_data, real_labels)]
#            data_mal[0] = poison_items(data_mal[0], num_poisoned)
            total_real = 0
            for j in range(num_permutations):
                add_real = sess.run(correct_prediction_distribution_classifier, feed_dict={x: data_real, y_: label_legitimate})
                data_real[0] = shuffle_inner_batch(data_real[0])
                if(add_real):
                    total_real += 1
            total_real_running += total_real
            total_mal = 0
            for j in range(num_permutations):
                add_mal = sess.run(correct_prediction_distribution_classifier, feed_dict={x: data_mal, y_: label_legitimate})
                data_mal[0] = shuffle_inner_batch(data_mal[0])
                if(add_mal):
                    total_mal += 1
            total_mal_running += total_mal
#            print(total_real)
#            print(total_mal)
#                return
            if(total_real >= num_permutations*acc_threshold):
                real_items_added += 1
#                if(legit_found):
#                    # wait until at least one legit batch that has been identified as legit has come up!
#                    found_at = []
#                    for index in range(16):
#                        new_batch = data_real
#                        new_batch[0] = replace_at_index(new_batch[0], legit_batch, index)
#                        add_real = sess.run(correct_prediction_distribution_classifier, feed_dict={x: new_batch, y_: label_legitimate})
#                        if(add_real):
#                            found_at.append(index)
#                    if(len(found_at) == 1):
#                        print("found_at")
#                        print(found_at)
#                        show_images(data_real)
#                        return
#            else:
#                legit_found = True
#                legit_batch = data_real[0]

            if(total_mal >= num_permutations*acc_threshold):
                mal_items_added += 1
#            if(add_real):
#                real_items_added += 1
#            if(add_mal):
#                mal_items_added += 1
        print(steps_needed)
        print(float(total_real_running)/(float(steps_needed) * float(num_permutations)))
        print(float(total_mal_running)/(float(steps_needed) * float(num_permutations)))
        print('Percent real added to classifier %0.3f' % (float(real_items_added)/float(steps_needed)))
        print('Percent mal added to classifier %0.3f' % (float(mal_items_added)/float(steps_needed)))
        print('Actual real added to classifier %d' % real_items_added)
        print('Actual mal added to classifier %d' % mal_items_added)

#        if(TRAIN_MNIST_CLASSIFIER):
            #Train for MNIST on seed+new data to see improvement

#        classes = [4, 5]
#        mnist = load_modified_mnist(classes)
#        test_data, test_labels = get_batch_of_batchs_validation(mnist, classes)
#        test_accuracy_temp = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
#        print('test set: accuracy on 4, 5 set: %0.3f' % test_accuracy_temp)

if __name__ == '__main__':
    tf.app.run(main=main)

