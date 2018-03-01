import os
import sys
import time
import argparse
import pprint

import tensorflow as tf
import numpy as np
from tf_net import tf_build_resnet
from tf_net import test_graph


from tf_attacks import rbg_to_grayscale, get_binary_secret, sign_term, corr_term, mal_data_synthesis, mal_data_synthesis2, set_params_init

from load_cifar import load_cifar
from hyperparams import *

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

FLAGS = tf.app.flags.FLAGS
#
#tf.app.flags.DEFINE_string('data-dir', os.getcwd() + '/dataset/',
#                           'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
#tf.app.flags.DEFINE_integer('max-steps', 60,
#                            'Number of mini-batches to train on. (default: %(default)d)')
#tf.app.flags.DEFINE_integer('log-frequency', 10,
#                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
#tf.app.flags.DEFINE_integer('save-model', 100,
#                            'Number of steps between model saves (default: %(default)d)')
#
## Optimisation hyperparameters
#tf.app.flags.DEFINE_integer('batch-size', 128, 'Number of examples per mini-batch (default: %(default)d)')
#tf.app.flags.DEFINE_float('learning-rate', 1e-4, 'Learning rate (default: %(default)d)')
#tf.app.flags.DEFINE_integer('img-width', 32, 'Image width (default: %(default)d)')
#tf.app.flags.DEFINE_integer('img-height', 32, 'Image height (default: %(default)d)')
#tf.app.flags.DEFINE_integer('img-channels', 3, 'Image channels (default: %(default)d)')
#tf.app.flags.DEFINE_integer('num-classes', 10, 'Number of classes (default: %(default)d)')
#tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
#                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
##tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')
#tf.app.flags.DEFINE_float('weight_decay', 0, '''scale for l2 regularization''')
#
#define log directory for tensorboard
log_dir = os.path.join(FLAGS.log_dir,'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))


#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
#import cifar10 as cf

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

def translate_images(in_images):
#changes size from (128, 3, 32, 32) to (128, 32, 32, 3)
#    return tf.transpose(in_images, [0, 2, 3, 1])
    return np.swapaxes(np.swapaxes(in_images, 1, 2), 2, 3)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                    crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]
        
        yield inp_exc, targets[excerpt]

#definitions

CAP = 'cap'  # Capacity abuse attack
COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
LSB = 'lsb'  # LSB encoding attack
NO = 'no'  # No attack

#run settings
# VALIDATION_FREQ = 100
CHECKPOINT_FREQ = 500
EPOCH_LIMIT = 20
PRINT_FREQ = 10

def reshape_data(X_train, y_train, X_test):
    # reshape train and subtract mean
    pixel_mean = np.mean(X_train, axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    X_train_flip = X_train[:, :, :, ::-1]
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    return X_train, y_train, X_test

def main(num_epochs=500, lr=0.1, attack=CAP, res_n=5, corr_ratio=0.0, mal_p=0.1):
    # Load the dataset
    pprint.pprint(locals(), stream=sys.stderr)
#    print(res_n)
#    if res_n != 1:
#        print("nope")
#        return
    res_n=1
#    num_epochs=1
    sys.stderr.write("Loading data...\n")
    X_train, y_train, X_test, y_test = load_cifar(10)
    
    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048], X_train[:, 2048:]))
    X_train = X_train.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))
    X_test = X_test.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
    
    mal_n = int(mal_p * len(X_train) * 2) #not sure what this does?
    n_out = len(np.unique(y_train)) #Or this?
    #from original code:
    if attack in {SGN, COR}:
        # get the gray-scaled data to be encoded
        raw_data = X_train if X_train.dtype == np.uint8 else X_train * 255
        if raw_data.shape[-1] != 3:
            raw_data = raw_data.transpose(0, 2, 3, 1)
        raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
        sys.stderr.write('Raw data shape {}\n'.format(raw_data.shape))
        hidden_data_dim = np.prod(raw_data.shape[1:])
    elif attack == CAP:
        hidden_data_dim = int(np.prod(X_train.shape[2:]))
        mal_n /= hidden_data_dim
        if mal_n == 0:
            mal_n = 1
        X_mal, y_mal, mal_n = mal_data_synthesis2(X_train, num_targets=mal_n)
        print(X_mal)
        print(y_mal)
        sys.stderr.write('Number of encoded image: {}\n'.format(mal_n))
        sys.stderr.write('Number of synthesized data: {}\n'.format(len(X_mal)))
    
    input_shape = (None, 3, X_train.shape[2], X_train.shape[3])

    X_train, y_train, X_test = reshape_data(X_train, y_train, X_test)
    X_val, y_val = X_test, y_test

#    input_var = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
#    input_var = tf.placeholder(tf.float32, [None, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])
    input_var = tf.placeholder(tf.float32, [None, X_train.shape[2], X_train.shape[3], 3])
#    input_var = tf.placeholder(tf.float32, [None, FLAGS.img_channels, FLAGS.img_width, FLAGS.img_height])
    y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    
    if attack == CAP:
        X_train_mal = np.vstack([X_train, X_mal])
        y_train_mal = np.concatenate([y_train, y_mal])
        X_train = X_train_mal
        y_train = y_train_mal
        X_train_mal = X_mal
        y_train_mal = y_mal

    net = tf_build_resnet(input_var=input_var, classes=n_out, input_shape=input_shape, n=res_n)
#    test_graph('logs2')
    #now the malicious data and the model have been generated, start the training process!

    n = len(X_train)
    print("ok then")
    sys.stderr.write("Number of training data, output: {}, {}...\n".format(n, n_out))

    params = tf.trainable_variables()
    print(params)
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in params])
    sys.stderr.write("Number of parameters in model: %d\n" % total_params)

    if attack == COR:
        n_hidden_data = total_params / int(hidden_data_dim)
        sys.stderr.write("Number of data correlated: %d\n" % n_hidden_data)
        corr_targets = raw_data[:n_hidden_data].flatten()
#        corr_targets = theano.shared(corr_targets) ???
        offset = set_params_init(params, corr_targets)
        corr_loss, r = corr_term(params, corr_targets, size=offset)
    elif attack == SGN:
        n_hidden_data = total_params / int(hidden_data_dim) / 8
        sys.stderr.write("Number of data sign-encoded: %d\n" % n_hidden_data)
        corr_targets = get_binary_secret(raw_data[:n_hidden_data])
#        corr_targets = theano.shared(corr_targets) ???
        offset = set_params_init(params, corr_targets)
        corr_loss, r = sign_term(params, corr_targets, size=offset)
    else:
        r = tf.constant(0., dtype=np.float32)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):



#    prediction = lasagne.layers.get_output(network)
    train_vars = tf.trainable_variables()
    WEIGHT_DECAY = 0.0000 # from original code
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars if 'bias' not in v.name ])
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net))
    total_loss = tf.reduce_mean(softmax_loss + lossL2 * WEIGHT_DECAY)
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
#    all_layers = lasagne.layers.get_all_layers(network)
#    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
#    loss += l2_penalty
    # add malicious term to loss function
    if attack in {SGN, COR}:
        corr_loss *= corr_ratio
        loss += corr_loss

    # save init
#    sh_lr = theano.shared(lasagne.utils.floatX(lr))

    #training settings
#    WEIGHT_DECAY = 0.0001 #from paper
    START_LEARNING_RATE = 0.01
    DECAY_STEPS = 100000  # decay the learning rate every 100000 steps
    DECAY_RATE = 0.1  # the base of our exponential for the decay
    MOMENTUM = 0.9 #from paper

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
#    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=sh_lr)
    decayed_learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate, momentum=MOMENTUM, use_nesterov=True).minimize(total_loss, global_step=global_step)
    actual_label = tf.argmax(y_, 1)
    prediction = tf.argmax(net, 1)
    is_correct = tf.equal(prediction, actual_label)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
    error = tf.subtract(1.0, accuracy)
#    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
#    test_loss = test_loss.mean()

#ignoring this?
    # As a bonus, also create an expression for the classification w:
#    if target_var.ndim == 1:
#        test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
#    else:
#        test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)

#    train_fn = theano.function([input_var, target_var], [loss, r], updates=updates)
#    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.

    #summaries added in
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    error_summary = tf.summary.scalar("Error", error)
    loss_summary = tf.summary.scalar("Loss", total_loss)
    l2_loss_summary = tf.summary.scalar("Loss L2", lossL2)

#    img_summary = tf.summary.image('Input Images', input_var, max_outputs=4)
#    aug_image_summary = tf.summary.image('Augmented Images', input_var, max_outputs=4)
#    test_img_summary = tf.summary.image('Test Images', input_var)

#    train_summary = tf.summary.merge([loss_summary, accuracy_summary, error_summary, l2_loss_summary, img_summary])
    train_summary = tf.summary.merge([loss_summary, accuracy_summary, error_summary, l2_loss_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary, error_summary])

    step = 0
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    sys.stderr.write("Starting training...\n")
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(log_dir + "_validation", sess.graph)
        step = 0
        epochs = range(1,EPOCH_LIMIT+1)
        for epoch in epochs:
            # shuffle training data
            train_indices = np.arange(n)
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices, :, :, :]
            y_train = y_train[train_indices]
            # In each epoch, we do a full pass over the training data:
            train_err_total = 0
            train_acc_total = 0
            train_batches = 0
            step = 0
            PRINT_FREQ = 100
            start_time = time.time()
            train_r = 0
#            for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True, augment=True):
#                inputs, targets = batch
#                step += 1
#                targets = translate_labels(targets)
#                inputs = translate_images(inputs)
##                print(targets)
##                err, r = train_fn(inputs, targets)
#                _, train_summary_str, train_acc, err, train_loss, l2_loss = sess.run([train_step, train_summary, accuracy,error, total_loss,lossL2],feed_dict={input_var: inputs, y_: targets})
#                if(step % PRINT_FREQ == 0):
#                    print('step {}, accuracy on training set : {}, loss: {}, L2 Loss: {}'.format(step, train_acc, train_loss, l2_loss))
#
##                train_r += r
#                train_acc_total += train_acc
#                train_err_total += err
#                train_batches += 1

            if attack == CAP:
                step = 0
                # And a full pass over the malicious data
                for batch in iterate_minibatches(X_train_mal, y_train_mal, 128, shuffle=True, augment=False):
                    step+= 1
                    inputs, targets = batch
                    targets = translate_labels(targets)
                    inputs = translate_images(inputs)
#                    err, r = train_fn(inputs, targets)
                    _, train_summary_str, train_acc, err, train_loss, l2_loss = sess.run([train_step, train_summary, accuracy, error, total_loss,lossL2],feed_dict={input_var: inputs, y_: targets})
                    if(step % PRINT_FREQ == 0):
                        print('step {}, accuracy on mal training set : {}, loss: {}, L2 Loss: {}'.format(step, train_acc, train_loss, l2_loss))
#                    train_r += r
                    train_acc_total += train_acc
                    train_err_total += err
                    train_batches += 1

            if attack == CAP:
                mal_err = 0
                mal_acc = 0
                mal_batches = 0
                step = 0
                for batch in iterate_minibatches(X_mal, y_mal, 600, shuffle=False):
                    inputs, targets = batch
                    step += 1
                    targets = translate_labels(targets)
                    inputs = translate_images(inputs)
#                    err, acc = val_fn(inputs, targets)

                    predictions_made = sess.run(net, feed_dict={input_var: inputs})
                    targets_new = []
                    predictions_new = []
                    for i in range(0, 200):
                        avg_pred = np.argmax(predictions_made[i*3]+predictions_made[i*3+1]+predictions_made[i*3+2])
#                        avg_pred = predictions_made[i*3]
#                        print(avg_pred.eval())
#                        avg_pred = tf.argmax(avg_pred, 1)
                        predictions_new.append(avg_pred)
                        targets_new.append(targets[i*3])

#                    prediction = tf.argmax(net, 1)

#                    predictions_new = translate_labels(predictions_new)
#                    print("predictions_new.eval()")
#                    print(predictions_new)
#
#                    print(tf.argmax(predictions_new, 1).eval())
#                    print("targets_new")
#                    print(tf.argmax(targets_new, 1).eval())
                    is_correct2 = tf.equal(tf.cast(predictions_new, tf.int64), tf.argmax(tf.convert_to_tensor(targets_new), 1))
#                    print("is_correct2.eval()")
#                    print(is_correct2.eval())
                    accuracy_mal = tf.reduce_mean(tf.cast(is_correct2, tf.float32)).eval()

                    validation_accuracy, err, summary_str = sess.run([accuracy, error,validation_summary], feed_dict={input_var: inputs, y_: targets})
                    if(step % (PRINT_FREQ/20) == 0):
                        print('step {}, accuracy on mal test set : {}/{}, err: {}'.format(step, accuracy_mal, validation_accuracy, err))
                    mal_err += err
#                    mal_acc += validation_accuracy
                    mal_acc += accuracy_mal
                    mal_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
##validation
#            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#                inputs, targets = batch
#                targets = translate_labels(targets)
#                inputs = translate_images(inputs)
##                err, acc = val_fn(inputs, targets)
#                validation_accuracy, err, summary_str = sess.run([accuracy, error, validation_summary], feed_dict={input_var: inputs, y_: targets})
#
#                val_err += err
#                val_acc += validation_accuracy
#                val_batches += 1

            # Then we sys.stderr.write the results for this epoch:
            sys.stderr.write("Epoch {} of {} took {:.3f}s\n".format(epoch, EPOCH_LIMIT, time.time() - start_time))
            sys.stderr.write("  training loss:\t\t{:.6f}\n".format(train_err_total / train_batches))
            sys.stderr.write("  training accuracy:\t\t{:.6f}\n".format(train_acc_total / train_batches))
            if attack == CAP:
                sys.stderr.write("  malicious loss:\t\t{:.6f}\n".format(mal_err / mal_batches))
                sys.stderr.write("  malicious accuracy:\t\t{:.2f} %\n".format(mal_acc / mal_batches))
            if attack in {SGN, COR}:
#                sys.stderr.write("  training r:\t\t{:.6f}\n".format(train_r / train_batches))
#                sys.stderr.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
                sys.stderr.write("  validation accuracy:\t\t{:.2f} %\n".format(validation_accuracy))
        # don't loop back when we reach the end of the test set
        
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            targets = translate_labels(targets)
            inputs = translate_images(inputs)
            test_accuracy_temp = sess.run([accuracy], feed_dict={input_var: X_test, y_: y_test})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
#            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count

        sys.stderr.write("Final results:\n")
        sys.stderr.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
        sys.stderr.write("  test accuracy:\t\t{:.2f} %\n".format(test_accuracy / 500 * 100))

if __name__ == '__main__':
    tf.app.run()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)    # learning rate
    parser.add_argument('--epoch', type=int, default=100)   # number of epochs for training
    parser.add_argument('--model', type=int, default=5)     # number of blocks in resnet
    parser.add_argument('--attack', type=str, default=CAP)  # attack type
    parser.add_argument('--corr', type=float, default=0.)   # malicious term ratio
    parser.add_argument('--mal_p', type=float, default=0.1) # proportion of malicious data to training
    args = parser.parse_args()
    #arg parsing is broke due to unknown tensorflow interaction, ????
    
    main(num_epochs=args.epoch, lr=args.lr, corr_ratio=args.corr, mal_p=args.mal_p, attack=args.attack, res_n=args.model)
