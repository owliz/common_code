import tensorflow as tf

file_nm = '../convae_data/tfrecord_dataset/avenue/clip_length_10/training_frames.tfrecords'
dataset = tf.data.TFRecordDataset(file_nm)

batch_size = 2
clip_length = 2

# def parse_example(serial_exmp):
#     # tf.FixedLenFeature([], tf.string)  # 0D, 标量
#     # tf.FixedLenFeature([3]volume...)   1D，长度为3
#     features={'volume': tf.FixedLenFeature([], tf.string),
#               'shape': tf.FixedLenFeature([3], tf.int64)}
#     parsed_features = tf.parse_single_example(serial_exmp, features)
#     volume = tf.decode_raw(parsed_features['volume'], tf.float16)
#     shape = tf.cast(parsed_features['shape'], tf.int32)
#     return volume, shape
def parse_example(serial_exmp):
    # tf.FixedLenFeature([], tf.string)  # 0D, 标量
    # tf.FixedLenFeature([3]volume...)   1D，长度为3
    features={'volume': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(serial_exmp, features)
    volume = tf.decode_raw(parsed_features['volume'], tf.float32)
    return volume
# 解析文件中所有记录
dataset = dataset.map(parse_example)

# 将数据集中的连续元素组成batch
dataset = dataset.batch(2)
# Repeats this dataset count times.
# Randomly shuffles the elements of this dataset.
# buffer_size 代表从原数据集中采样的数量
# 维持一个buffer size 大小的 shuffle buffer，图中所需的每个样本从shuffle buffer中获取，
# 每取出一个样本，就从原数据集中重采样一个加入shuffle buffer中且会再shuffle一次。
dataset = dataset.shuffle(buffer_size=batch_size)
dataset = dataset.repeat(count=6)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(10):
        volume = sess.run(next_element)
        volume = tf.reshape(volume, [batch_size, clip_length, 227, 227])
        volume = tf.transpose(volume, [0, 2, 3, 1])
        import matplotlib
        import matplotlib.pyplot as plt
        sample = tf.transpose(volume[0], [2, 0, 1]).eval()
        from skimage import io
        for i in range(sample.shape[0]):
            # 转回float32
            img = sample[i].astype('float32')
            io.imshow(img)
        io.show()
            # io.imsave('cat.jpg', sample[i])

# iterator = dataset.make_one_shot_iterator()


