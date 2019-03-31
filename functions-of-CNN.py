
"""定义了卷积层、池化层和全连接层实现的方法函数"""

import tensorflow as tf

# 卷积的方法
def conv(self, name, input_data, out_channel):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weight', [3, 3, in_channel, out_channel], dtype=tf.float32)
        biases = tf.get_varlable('biases', [out_channel], dtype=tf.float32)
        conv_res = tf.nn.conv2d(input_data, kernel, [1,1,1,1], padding='SAME')
        res = tf.nn.bias_add(conv_res, biases)
        out = tf.nn.relu(res, name=name)
    return out

# 全连接层的方法
def fc(self, name, input_data, out_channel):
    shape = input_data.get_shape().as_list()
    # size定义了作为输入的神经元的个数
    # 如果输入是[batch,width,height,channels]，则size = w * h * c
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    # 将输入数据展开成一维列向量
    input_data_flat = tf.reshape(input_data, [-1, size])
    with tf.variable_scope(name):
        # 其中，size为作为输入的神经元的个数
        weights = tf.get_variable(name='weight', shape=[size, out_channel], dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[out_channel], dtype=tf.float32)
        res = tf.matmul(input_data_flat, weights)
        out = tf.nn.relu(tf.nn.bias_add(res, biases))
    return out

# 池化层的方法
def max_pool(self, name, input_data):
    out = tf.nn.max_pool(input_data, [1,2,2,1], [1,2,2,1], padding='SAME', name=name)
    return out

