
"""定义VGG-16模型类：VGG-16的tensorflow实现"""
import tensorflow as tf
import numpy as np

"""
微调VGG16模型：全连接层的神经元个数，trainable参数
1. 预训练的VGG是在ImageNet数据集上进行的，对1000个类别进行判定，若希望利用已训练模型用于其他分类任务，
需要修改最后的全连接层（猫狗大战为2分类，所以改为2层）；同时该层参数需要训练，trainable=true
2. 在进行微调对模型进行重新训练时，对于部分不需要训练的层可以通过设置trainable=False，来确保其在训练
过程中不会被修改权值
"""

class Vgg16:
    # 初始化
    def __init__(self, imgs):
        self.parameters = [] # 在类的初始化时加入全局列表，将所需共享的参数加载进来
        self.imgs = imgs
        self.convlayers()    # 模型定义
        self.fc_layers()     # 模型定义
        self.probs = tf.nn.softmax(self.fc8)   # 模型输出:输出属于各个类别的概率值

    def saver(self):   # 对复用类的定义没有影响
        return tf.train.Saver()

    # 定义池化层方法：没有相应参数（不涉及微调与否的事）
    def maxpool(self, name, input_data):
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)
        return out

    # 定义卷积方法
    def conv(self, name, input_data, out_channel, trainable=False):   # 设置trainable为False的相应参数不参加训练
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=False)
            biases = tf.get_variable('biases', [out_channel], dtype=tf.float32 ,trainable=False)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding='SAME')
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]  # 将卷积层定义的参数(kernel,biases)加入全局列表
        return out

    # 定义全连接层
    def fc(self, name, input_data, out_channel, trainable=True): # trainable参数设为true,表示全连接层参数需要参与训练
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name='weights', shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name='biases', shape=[out_channel], dtype=tf.float32, trainable=trainable)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        self.parameters += [weights, biases] # 将全连接层定义的参数(weights,biases)加入全局参数列表
        return out

    # 堆叠起来的卷积层和池化层结构:CNN层的所有参数都不需要调整，trainable设为False
    def convlayers(self):
        # conv1
        self.conv1_1 = self.conv('conv1re_1', self.imgs, 64, trainable=False)   # self.imgs原始数据输入
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, 64, trainable=False)
        self.pool1 = self.maxpool('poolre1', self.conv1_2)

        # conv2
        self.conv2_1 = self.conv('conv2_1', self.pool1, 128, trainable=False)
        self.conv2_2 = self.conv('convwe2_2', self.conv2_1, 128, trainable=False)
        self.pool2 = self.maxpool('pool2', self.conv2_2)

        # conv3
        self.conv3_1 = self.conv('conv3_1', self.pool2, 256, trainable=False)
        self.conv3_2 = self.conv('convrwe3_2', self.conv3_1, 256, trainable=False)
        self.conv3_3 = self.conv('convrew3_3', self.conv3_2, 256, trainable=False)
        self.pool3 = self.maxpool('poolre3', self.conv3_3)

        # conv4
        self.conv4_1 = self.conv('conv4_1', self.pool3, 512, trainable=False)
        self.conv4_2 = self.conv('convrwe4_2', self.conv4_1, 512, trainable=False)
        self.conv4_3 = self.conv('conv4rwe_3', self.conv4_2, 512, trainable=False)
        self.pool4 = self.maxpool('pool4', self.conv4_3)

        # conv5
        self.conv5_1 = self.conv('conv5_1', self.pool4, 512, trainable=False)
        self.conv5_2 = self.conv('convrwe5_2', self.conv5_1, 512, trainable=False)
        self.conv5_3 = self.conv('conv5_3', self.conv5_2, 512, trainable=False)
        self.pool5 = self.maxpool('poolrwe5', self.conv5_3)

    # 堆叠起来的全连接层结构
    def fc_layers(self):
        self.fc6 = self.fc('fc1', self.pool5, 4096, trainable=False)
        self.fc7 = self.fc('fc2', self.fc6, 4096, trainable=False)
        self.fc8 = self.fc('fc3', self.fc7, 2, trainable=True)   # 输出类别的个数（猫狗大战案例输出类别是2）
        # fc8正是我们案例中需要训练的，因此trainable设为True

    # 复用已经训练好的权重参数，通过load方法将数据以字典的形式读入
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        print(weights)
        keys = sorted(weights.keys())
        print(keys)
        for i,k in enumerate(keys):   # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            if i not in [30, 31]: # 剔除不需要载入的层(fc8和fc8的输出）
                sess.run(self.parameters[i].assign(weights[k]))
        print('------------------Weights Loaded!----------------------')