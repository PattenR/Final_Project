import tensorflow as tf
import os
import numpy as np

#resnet goes here!
#based on https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
#based on original ml models that remember to much

FLAGS = tf.app.flags.FLAGS
BN_EPSILON = 0.001
def activation_summary(x):
    '''
        :param x: A Tensor
        :return: Add histogram summary and scalar summary of the sparsity of the tensor
        '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    #can add regularisation here, need to change the flag so the weight decay is more than 0 if I want to do that, and add to the new_variables the regualiser.
#    if is_fc_layer is True:
#        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
#    else:
#        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

#    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
#                                    regularizer=regularizer)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
        :return: output layer Y = WX + B
        '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
                            
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
                           
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''
    
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
    
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)
    
    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''
    
    in_channel = input_layer.get_shape().as_list()[-1]
    
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(l, increase_dim=False, first_block=False ,projection=True):
#    input_num_filters = l.output_shape[1]
#    input_channel = l.get_shape().as_list()[-1]
#    print(l.get_shape().as_list())
    input_channel = l.get_shape().as_list()[3]
#    input_channel = l.output_shape[1]
    if increase_dim:
#        first_stride = (2, 2)
        stride = 2
        output_channel = input_channel*2
#        out_num_filters = input_num_filters * 2
    else:
#        first_stride = (1, 1)
        stride = 1
        output_channel = input_channel
#        out_num_filters = input_num_filters

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(l, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(l, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

#    stack_1 = batch_norm(Conv2DLayer(l, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
#
#    stack_2 = batch_norm(Conv2DLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # add shortcut connections
    if increase_dim:
#        if projection:
#            # projection shortcut, as option B in paper
#            projection = batch_norm(
#            Conv2DLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
#            pad='same', b=None, flip_filters=False))
#            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
#        else:
#            # identity shortcut, as option A in paper
#            identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
#            padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
#            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        pooled_input = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
#        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)
        padded_input = l
    output = conv2 + padded_input
    return output

def tf_build_resnet(input_var=None, input_shape=(None, 3, 50, 50), n=5, classes=10, reuse=False):
    #reuse basically the same as train/test?
    #default train
    #needs softmax added
    layers = []
    input_shape = [3, 3, 3, 16]
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_var, input_shape, 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], first_block=True)
                print(conv1.get_shape())
            else:
                conv1 = residual_block(layers[-1])
            print(conv1.get_shape())
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            if i == 0:
                conv2 = residual_block(layers[-1], increase_dim=True)
                print(conv2.get_shape())
            else:
                conv2 = residual_block(layers[-1])
                print(conv2.get_shape())
            activation_summary(conv2)
            layers.append(conv2)
    
    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            if i == 0:
                conv3 = residual_block(layers[-1], increase_dim=True)
                print(conv3.get_shape())
            else:
                conv3 = residual_block(layers[-1])
                print(conv3.get_shape())
            layers.append(conv3)
#        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        print(bn_layer.get_shape())
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])
        
#        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        print(output.get_shape())
        layers.append(output)

    return layers[-1]

def test_graph(train_dir='logs'):
    '''
        Run this function to look at the graph structure on tensorboard. A fast way!
        :param train_dir:
        '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)

    result = tf_build_resnet(input_tensor)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

