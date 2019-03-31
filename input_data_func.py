import os
import numpy as np
import tensorflow as tf

def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
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
    print(type(image_list))

    return image_list, label_list


from vgg_preprocess import preprocess_for_train
# 图像预处理模块vgg_preprocess：为了保持与vggnet训练时图像预处理一样的方式，对本案例中的图像进行预处理

# vggnet模型的图片大小是224*224，alexnet模型图片大小是227*227
img_width = 224
img_height = 224

def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity): # 通过读取列表批量载入图形和标签
    # capacity参数：内存中存储的最大数据容量，可以根据自己的硬件配置来指定
    # image = tf.cast(image_list, tf.string)
    image = tf.as_string(image_list, tf.string)
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

