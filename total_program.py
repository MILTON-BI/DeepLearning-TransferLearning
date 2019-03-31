import os
import tensorflow as tf
import numpy as np
from time import time
import VGG16_model as model
import input_data_func as idf


#-------------------------------------迭代训练-----------------------------------------------
startTime = time()
batch_size = 32   # 每批处理的样本数量
capacity = 256  # 内存中存储的最大数据容量
means = [123.68, 116.779, 103.939] # vgg训练时图像预处理所减均值（RGB三通道）

xs, ys = idf.get_file('./data/train/*')

image_batch, label_batch = idf.get_batch(xs, ys, 224, 224, batch_size, capacity) # 通过调用get_batch函数载入图像和标签

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int32, [None, 2])

# 微调finetuining
vgg = model.vgg16(x)
fc8_finetuining = vgg.probs # 即softmax(fc8)

# 定义损失函数和优化器
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8_finetuining, labels=y))
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
    labels = idf.onehot(labels)  # 用one-hot形式对标签进行编码

    sess.run(optimizer, feed_dict={x: images, y: labels})
    loss = sess.run(loss_func, feed_dict={x: images, y: labels})
    print('当前损失值为：%f' % loss)

    epoch_end_time = time()
    print('当前轮次耗时：', (epoch_end_time - epoch_start_time))
    epoch_start_time = epoch_end_time

    if (i+1) % 500 ==0:
        saver.save(sess, os.path.join('./model/', 'epoch {:06d}.ckpt'.format(i)))
    print('--------------------%d轮次完成-------------------------' % i)

# 模型保存
saver.save(sess, './model/')
print('-----------------优化完成！------------------')

# 输出优化耗费时间
duration = time() - startTime
print('------------------训练完成！总耗时:{:.2f}--------------------'.format(duration))

# 关闭线程
coord.request_stop()  # 通知其他线程关闭
coord.join(threads)  # join操作等待其他线程结束，其他所有线程关闭后，这一函数才能返回
sess.close()