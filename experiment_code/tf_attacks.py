import sys
import numpy as np
import tensorflow as tf

#from original
def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])
#original
def get_binary_secret(X):
    # convert 8-bit pixel images to binary with format {-1, +1}
    assert X.dtype == np.uint8
    s = np.unpackbits(X.flatten())
    s = s.astype(np.float32)
    s[s == 0] = -1
    return s

#translated from theano
def corr_term(params, targets, size=None):
    # malicious term that maximizes correlation between targets and params
    # x should a vector of floating point numbers
    
    if isinstance(params, list):
        params = tf.concat([tf.contrib.layers.flatten(p) for p in params if p.ndim > 1])
    
    if size is not None:
        targets = targets[:size]
        params = params[:size]
        sys.stderr.write('Number of parameters correlated {}\n'.format(size))
    
    # calculate Pearson's correlation coefficient using tensorflow
    p_mean = tf.reduce_mean(params)
    t_mean = tf.reduce_mean(targets)
    p_m = params - p_mean
    t_m = targets - t_mean
    r_num = tf.cast(tf.reduce_sum(p_m * t_m), tf.float64)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(p_m, tf.float64))) * tf.reduce_sum(tf.square(tf.cast(t_m, tf.float64))))
    r = r_num / r_den
    loss = abs(r)
    return - loss, r

#translated from theano
def sign_term(params, targets, size):
    # malicious term that penalizes sign mismatch between x and params
    # x should a binary (+1, -1) vector
    if isinstance(params, list):
        params = tf.concat([tf.contrib.layers.flatten(p) for p in params if p.ndim > 1])

    sys.stderr.write('Number of parameters correlated {}\n'.format(size))

    targets = tf.contrib.layers.flatten(targets) #maybe broken ?
    targets = targets[:size]
    params = params[:size]

    # element-wise multiplication
    constraints = targets * params
    penalty = tf.case(tf.greater(constraints, 0), 0, constraints)
    penalty = abs(penalty)
    correct_sign = tf.reduce_mean(tf.greater(constraints, 0))
    return tf.reduce_mean(penalty), correct_sign

#original
def mal_data_synthesis(train_x, num_targets=10, precision=4):
    # synthesize malicious images to encode secrets
    # for CIFAR, use 2 data points to encode one approximate 4-bit pixel
    # thus divide the number of targets by 2
#    print("mal synth\n")

    num_targets /= 2
    if num_targets == 0:
        num_targets = 1
#    print(train_x)
    targets = train_x[:num_targets]
#    print(targets)
    input_shape = train_x.shape
    if input_shape[1] == 3:     # rbg to gray scale
        targets = rbg_to_grayscale(targets.transpose(0, 2, 3, 1))

    mal_x = []
    mal_y = []
    for j in range(num_targets):
#        target = targets[j].flatten
        target = targets[j].flatten()
#        print("target")
#        print(target)
#        print(target.size)
        for i, t in enumerate(target):
            t = int(t * 255)
#            if(t != 0):
#                print(t)
#            print("t")
#            print(t)
            # get the 4-bit approximation of 8-bit pixel
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            # use 2 data points to encode p
            # e.g. pixel=15, use (x1, 7), (x2, 8) to encode
            p_bits = [p / 2, p - p / 2]
#            print("p_bits")
#            print(p_bits)
            for k, b in enumerate(p_bits):
#                print(k)
                # initialize a empty image
#                x = np.zeros(input_shape[1:]).reshape(3, -1)
                x = np.zeros(input_shape[1:])
                # simple & naive deterministic value for two pixel
                channel = j % 3
                value = j / 3 + 1.0
                x[i] = value
                if i < len(target) - 1:
                    x[i + 1] = k + 1.0
                else:
                    x[0] = k + 1.0 + channel 
            
#                print("x")
#                print(x)
#                print("b")
#                print(b)
                mal_x.append(x)
                mal_y.append(b)
#            if(i > 152):
#                for x  in range(i):
#                    print(mal_x[x])
#                for i in range(10):
#                    print("i reached")
#                print(mal_x)
#                print(mal_y)
 #               break

    mal_x = np.asarray(mal_x, dtype=np.float32)
    mal_y = np.asarray(mal_y, dtype=np.int32)
#    shape = [-1] + list(input_shape[1:])
#    mal_x = mal_x.reshape(shape)
#    print("mal_x size")
#    print(mal_x.shape)
#    print(mal_x)
    return mal_x, mal_y, num_targets
#maybe come back to this
#def get_deterministic_image(image_number, total_images, aug_num, im_shape_1, im_shape_2):
## need to generate an  im_shape_1xim_shape_2 images from image number and total images
#    image_frac = (float(image_number)/float(total_images))
#    total_pix = im_shape_1*im_shape_2
#    center_pix = image_frac*float(total_pix)
#    x = np.zeros(im_shape_1, im_shape_2)




#modified version that returns multiple synth images per encoding
def mal_data_synthesis2(train_x, num_targets=10, precision=4):
    # synthesize malicious images to encode secrets
    # for CIFAR, use 2 data points to encode one approximate 4-bit pixel
    # thus divide the number of targets by 2
    num_targets /= 2
    if num_targets == 0:
        num_targets = 1
    
    targets = train_x[:num_targets]
    input_shape = train_x.shape
    if input_shape[1] == 3:     # rbg to gray scale
        targets = rbg_to_grayscale(targets.transpose(0, 2, 3, 1))
    
    mal_x = []
    mal_y = []
    fraction = float(1)/float(num_targets)
    for j in range(num_targets):
        target = targets[j].flatten()
        for i, t in enumerate(target):
            print("t")
            print(t)
            print("i")
            print(i)
            t = int(t * 255)
            # get the 4-bit approximation of 8-bit pixel
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            # use 2 data points to encode p
            # e.g. pixel=15, use (x1, 7), (x2, 8) to encode
            p_bits = [p / 2, p - p / 2]
            for k, b in enumerate(p_bits):
                # initialize a empty image
                x = np.zeros(input_shape[1:]).reshape(3, -1)
                x2 = np.zeros(input_shape[1:]).reshape(3, -1)
                x3 = np.zeros(input_shape[1:]).reshape(3, -1)
#                x = get_deterministic_image(j*len(target)+i*(len(p_bits)+k), (num_targets+1)*len(target), aug_num, 3, len(target))
                # simple & naive deterministic value for two pixel
                channel = j % 3
                value = j / 3 + 1.0
                print("channel")
                print(channel)
                print("value")
                print(value)
#                x[channel, i] = (float(value)/2 - float(1)/float(4))*(fraction*j)

                x[channel, i] = value
                x2[channel, i] = value
                if i < len(target) - 1:
                    x[channel, i + 1] = k + 1.0
                    x3[channel, i + 1] = k + 1.0
#                    x[channel, i+1] = (float(k + 1.0)/2 - float(1)/float(4))*(fraction*j)
                else:
                    x[channel, 0] = k + 1.0
                    x3[channel, 0] = k + 1.0
#                    x[channel, 0] = (float(k + 1.0)/2 - float(1)/float(4))*(fraction*j)

                if i-1 > 0:
                    x2[channel, i - 1] = k + 1.0
                    x3[channel, i - 1] = k + 1.0
                else:
                    x2[channel, 0] = k + 1.0
                    x3[channel, 0] = k + 1.0

                mal_x.append(x)
                mal_y.append(b)
                mal_x.append(x2)
                mal_y.append(b)
                mal_x.append(x3)
                mal_y.append(b)

    mal_x = np.asarray(mal_x, dtype=np.float32)
    mal_y = np.asarray(mal_y, dtype=np.int32)
    shape = [-1] + list(input_shape[1:])
    mal_x = mal_x.reshape(shape)
    return mal_x, mal_y, num_targets


def set_params_init(params, values, num_param_to_set=60):
    # calculate number of parameters needed to encode secrets
    
    if not isinstance(values, np.ndarray):
        values = values.get_value()
    
    params_to_set = []
    for p in params:
        if p.ndim > 1:
            params_to_set.append(p)
        if len(params_to_set) >= num_param_to_set:
            break

    offset = 0
    for p in params_to_set:
        shape = p.get_value().shape
        n = np.prod(shape)
        if offset + n > len(values):
            offset = len(values)
            sys.stderr.write('Number of params greater than targets\n')
            break
        offset += n

    return offset

#sign_term(np.array([1,2,3]), np.array([1,2,3]), size=None)

