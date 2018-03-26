from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import tensorflow as tf
import sys
import numpy as np
from cifar10_input import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
import cifar10 as cf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '{cwd}/logs_reduced/'.format(cwd=os.getcwd()),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('savemodel', 1000,
                            'Number of steps between model saves (default: %(default)d)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_string('datadir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')

tf.app.flags.DEFINE_string('logdir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')

run_log_dir = os.path.join(FLAGS.logdir,
                           'exp_bs_{bs}_CNN'.format(bs=128))

# Global constants describing the CIFAR-10 data set.

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def deepCNN(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  print("images.get_shape()")
  print(images.get_shape())
  
  images = tf.reshape(images, [-1, 32, 32, 3])
  
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = bias_variable([64])
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
#    _activation_summary(conv1)

  print("conv1.get_shape()")
  print(conv1.get_shape())
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1],
                         padding='SAME', name='pool1')
  print("pool1.get_shape()")
  print(pool1.get_shape())
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  print("norm1.get_shape()")
  print(norm1.get_shape())

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
#    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    biases = bias_variable([64])
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
#    _activation_summary(conv2)

  print("conv2.get_shape()")
  print(conv2.get_shape())

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  print("norm2.get_shape()")
  print(norm2.get_shape())
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')


#  (?, 6, 6, 64)
#  (?, 6, 6, 8)
  #dimensionallity reduction step

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
     shape=[5, 5, 64, 64],
     stddev=5e-2,
     wd=None)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
#    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    biases = bias_variable([64])
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
#    _activation_summary(conv3)
  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm2')
  print("norm3.get_shape()")
  print(norm3.get_shape())
  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  print("pool3.get_shape()")
  print(pool3.get_shape())

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
#    tensor = tf.convert_to_tensor([images.get_shape()[0], -1])
    reshape = tf.reshape(pool3, [-1, 256])
#    reshape = tf.reshape(conv3, [images.get_shape()[0], -1])
    print(reshape.get_shape())
    dim = reshape.get_shape()[1].value
    print('DIM')
    print(dim)
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
#    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    biases = bias_variable([384])
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
#    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    biases = bias_variable([192])
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
#    biases = _variable_on_cpu('biases', [NUM_CLASSES],
#                              tf.constant_initializer(0.0))
    biases = bias_variable([NUM_CLASSES])
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#    _activation_summary(softmax_linear)

  return softmax_linear, pool3

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
        
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
    decay is not added for this Variable.
    
    Returns:
    Variable Tensor
    """
#    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#    var = _variable_on_cpu(
#       name,
#       shape,
#       tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    initial = tf.constant(stddev, shape=shape)
    var = tf.Variable(initial, name=name)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def main(argv=None):  # pylint: disable=unused-argument
#  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  tf.reset_default_graph()
      # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / 128
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  
  # Import data
  cifar = cf.cifar10(batchSize=128, downloadDir=FLAGS.datadir)

  with tf.variable_scope('inputs'):
      # Create the model
      x = tf.placeholder(tf.float32, [None, 3072])
      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, CNN_final = deepCNN(x)
  global_step = tf.train.get_or_create_global_step()
  with tf.variable_scope('x_entropy'):
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    global_step,
    decay_steps,
    LEARNING_RATE_DECAY_FACTOR,
    staircase=True)
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
  loss_summary = tf.summary.scalar('Loss', cross_entropy)
  acc_summary = tf.summary.scalar('Accuracy', accuracy)

  finalRepresentations = []

  # summaries for TensorBoard visualisation
  validation_summary = tf.summary.merge([acc_summary])
  training_summary = tf.summary.merge([loss_summary])
  test_summary = tf.summary.merge([acc_summary])

  # saver for checkpoints
  saver = tf.train.Saver(max_to_keep=1)

  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
      summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)
      
      sess.run(tf.global_variables_initializer())
      k = tf.placeholder(tf.float32)
      # writer = tf.summary.FileWriter(os.getcwd() + "/data/histogram_example")
      summaries = tf.summary.merge_all()

# Training and validation
      for step in range(FLAGS.max_steps):
          # Training: Backpropagation using train set
#          (trainImages, trainLabels) = cifar.getTrainBatch()
          (trainImages, trainLabels) = distorted_inputs(FLAGS.datadir, 128, cifar)
#          (testImages, testLabels) = cifar.getTestBatch()
          (testImages, testLabels) = inputs(False, FLAGS.datadir, 128, cifar)

#          print(trainLabels.shape)
#          print(trainImages.shape)
          _, summary_str = sess.run([train_step, training_summary], feed_dict={x: trainImages, y_: trainLabels})

#            summ = sess.run(summaries, feed_dict={k: k_val})

          # writer.add_summary(summ, global_step=step)

          if step % (FLAGS.log_frequency + 1)== 0:
              summary_writer.add_summary(summary_str, step)

          # Validation: Monitoring accuracy using validation set
          if step % FLAGS.log_frequency == 0:
              validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: testImages, y_: testLabels})
              print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
              summary_writer_validation.add_summary(summary_str, step)

          # Save the model checkpoint periodically.
          if step % FLAGS.savemodel == 0 or (step + 1) == FLAGS.max_steps:
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
  tf.app.run()
