import os
import tensorflow as tf
import numpy as np
from time import time
import VGG16_model as model
from vgg_preprocess import preprocess_for_train
# 图像预处理模块vgg_preprocess：为了保持与Vggnet训练时图像预处理一样的方式，对本案例中的图像进行预处理
# from input_data_func import *

def get_file(file_dir):
    images = []
    temp = []
    labels = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list




# vggnet模型的图片大小是224*224，alexnet模型图片大小是227*227
img_width = 224
img_height = 224

def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity): # 通过读取列表批量载入图形和标签
    # capacity参数：内存中存储的最大数据容量，可以根据自己的硬件配置来指定
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = preprocess_for_train(image, 224, 224)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

# 对标签形式进行重构，转换成独热编码
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

startTime = time()
batch_size = 32   # 每批处理的样本数量
capacity = 256  # 内存中存储的最大数据容量
means = [123.68, 116.779, 103.939] # vgg训练时图像预处理所减均值（RGB三通道）

xs, ys = get_file('data/train/')
# print(xs)
# print(ys)
image_batch, label_batch = get_batch(xs, ys, 224, 224, batch_size, capacity) # 通过调用get_batch函数载入图像和标签
# print(image_batch)
# print(label_batch)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 2])

# 微调finetuining
vgg = model.Vgg16(x)
fc8_finetuining = vgg.probs # 即softmax(fc8)

# 定义损失函数和优化器
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining, labels=y))
# correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y, tf.float32))
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_func)

# 启动会话，进行训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
vgg.load_weights('./vgg16/vgg16_weights.npz', sess) # 读取已经训练好vgg模型相应的权重参数，将权重载入以实现复用
saver = tf.train.Saver()

# 启动线程
coord = tf.train.Coordinator()  # 使用协调器coordinator来管理线程
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

epoch_start_time = time()

# 迭代训练
for i in range(1000):
    images, labels = sess.run([image_batch, label_batch])
    labels = onehot(labels)  # 用one-hot形式对标签进行编码
    sess.run(optimizer, feed_dict={x: images, y: labels})
    loss = sess.run(loss_func, feed_dict={x: images, y: labels})
    print('当前损失值为：%f' % loss)

    epoch_end_time = time()
    print('当前轮次耗时：', (epoch_end_time - epoch_start_time))
    epoch_start_time = epoch_end_time

    if (i+1) % 50 ==0:
        saver.save(sess, os.path.join('model/', 'epoch {:06d}.ckpt'.format(i)))
    print('--------------------%d轮次完成-------------------------' % i)

# 模型保存
saver.save(sess, 'model/')
print('-----------------优化完成！------------------')

# 输出优化耗费时间
duration = time() - startTime
print('------------------训练完成！总耗时:{:.2f}--------------------'.format(duration))

# 关闭线程
coord.request_stop()  # 通知其他线程关闭
coord.join(threads)  # join操作等待其他线程结束，其他所有线程关闭后，这一函数才能返回