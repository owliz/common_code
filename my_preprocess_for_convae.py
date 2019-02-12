import logging
import coloredlogs
import datetime
import os
import uuid
from skimage.io import imread
import numpy as np
import tensorflow as tf
import sys



resize_height = 224
resize_width = 224

device = 'gpu0'

job_uuid = str(uuid.uuid4())
log_path = os.path.join('logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_path,
                                          "{}.log".format(datetime.datetime.now().strftime(
                                              "%Y%m%d-%H%M%S"))),
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
coloredlogs.install(level=logging.INFO)
logger = logging.getLogger()


if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.debug("Using CPU only")
elif device == 'gpu0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.debug("Using GPU 0")


def to_npy(dataset, dataset_path, frames_type='training_frames'):
    """
    transfer frames into grey, then into npy.
    :param dataset: dataset name
    :param dataset_path: dataset root dir
    :return: .npy  file
    """
    logger.info("[{}] to npy for [{}]".format(frames_type, dataset))
    frame_path = os.path.join(dataset_path, dataset, frames_type)
    for frames_folder in os.listdir(frame_path):
        print('==> ' + os.path.join(frame_path, frames_folder))
        training_frames_vid = []
        for frame_file in sorted(os.listdir(os.path.join(frame_path, frames_folder))):
            frame_file_name = os.path.join(frame_path, frames_folder, frame_file)
            # 灰度 [-1, 1]
            frame_value = imread(frame_file_name,as_grey=True)
            assert(0. <= frame_value.all() <= 1.)
            training_frames_vid.append(frame_value)
        training_frames_vid = np.array(training_frames_vid)

        os.makedirs(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type), exist_ok=True)
        np.save(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                             '{}_{}.npy'.format(frames_type, frames_folder)),
                training_frames_vid)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def volumes_counter(dataset, dataset_path, frames_type='training_frames', clip_length=10, stride=1):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    print("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type)))
    tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset,
                                'clip_length_{:02d}'.format(clip_length))
    os.makedirs(tfrecord_dir, exist_ok=True)

    all_vol_num = 0
    for i in range(num_videos):
        data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                                           '{}_{:02d}.npy'.format(frames_type, i+1)))
        num_frames = data_frames.shape[0]
        vol_num = num_frames - ((clip_length-1)*stride + 1) + 1
        all_vol_num += vol_num

    save_path = os.path.join(tfrecord_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(frames_type,
                                                                                       stride))
    np.savetxt(save_path, np.array([all_vol_num + 1]), fmt='%d')
    print('volumes num is [{}] for [stride_{}]'.format(all_vol_num + 1, stride))
    logger.info('volumes num is [{}] for [stride_{}]'.format(all_vol_num + 1, stride))


def to_tfrecord(dataset, dataset_path, frames_type='training_frames', clip_length=10, stride=1):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    print("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}]".format(stride, frames_type, dataset))
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type)))
    tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset,
                                'clip_length_{:02d}'.format(clip_length))
    os.makedirs(tfrecord_dir, exist_ok=True)
    desfile = os.path.join(tfrecord_dir, '{}_stride_{}.tfrecords'.format(frames_type, stride))

    with tf.python_io.TFRecordWriter(desfile) as writer:
        all_vol_num = 0
        for i in range(num_videos):
            data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                                               '{}_{:02d}.npy'.format(frames_type, i+1)))
            # 末尾增加一个维度
            # data_frames = np.expand_dims(data_frames, axis=-1)
            num_frames = data_frames.shape[0]
            vol = 0
            vol_num = num_frames - ((clip_length-1)*stride + 1) + 1
            volumes = np.zeros((vol_num, clip_length, resize_height, resize_width)).astype('float32')

            for j in range(vol_num):
                volumes[vol] = data_frames[j:j + (clip_length - 1) * stride + 1:stride]
                saved_volume = volumes[vol]
                # Create an example protocol buffer
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'volume': _bytes_feature(saved_volume)}
                    )
                )
                # example = tf.train.Example(
                #     features=tf.train.Features(
                #         feature={'volume': _bytes_feature(saved_volume),
                #                  'shape': _int64_feature(saved_volume.shape)}
                #     )
                # )
                writer.write(example.SerializeToString())
            all_vol_num += vol_num

    print("to_volume_stride_[{}] for [{}] [{}] finished".format(stride, frames_type, dataset))
    logger.info("to_volume_stride_[{}] for [{}] [{}] finished".format(stride, frames_type, dataset))
    save_path = os.path.join(tfrecord_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(frames_type,
                                                                                       stride))
    np.savetxt(save_path, np.array([vol_num + 1]), fmt='%d')
    print('[{}] volumes are saved'.format(vol_num + 1))
    logger.info('[{}] volumes are saved'.format(vol_num + 1))


def to_tfrecord_split(dataset, dataset_path, frames_type='training_frames', clip_length=10,
                      stride=1, validate_split=0.15):
    """
    transfer to volume unit and saved in tfrecord
    :param dataset:
    :param dataset_path:
    :param frames_type:
    :param clip_length:
    :return:
    """
    num_videos = len(os.listdir(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type)))
    tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset,
                                'clip_length_{:02d}'.format(clip_length))
    os.makedirs(tfrecord_dir, exist_ok=True)
    desfile_1 = os.path.join(tfrecord_dir, 'train_stride_{}_validate_split_{}.tfrecords'.format(
        stride, validate_split))
    desfile_2 = os.path.join(tfrecord_dir, 'val_stride_{}_validate_split_{}.tfrecords'.format(
        stride, validate_split))

    save_path = os.path.join(tfrecord_dir, 'num_of_volumes_in_{}_stride_{}.txt'.format(
        frames_type, stride))
    all_vol_num = np.loadtxt(save_path, dtype=int)

    train_size = int((1 - validate_split)*all_vol_num)

    write_1 = tf.python_io.TFRecordWriter(desfile_1)
    write_2 = tf.python_io.TFRecordWriter(desfile_2)
    count = 0
    for i in range(num_videos):
        data_frames = np.load(os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                                           '{}_{:02d}.npy'.format(frames_type, i+1)))
        # 末尾增加一个维度
        # data_frames = np.expand_dims(data_frames, axis=-1)
        num_frames = data_frames.shape[0]
        volumes = np.zeros((num_frames-clip_length+1, clip_length, resize_height,
                            resize_width)).astype('float32')
        vol = 0
        vol_num = num_frames - ((clip_length - 1) * stride + 1) + 1

        for j in range(vol_num):
            volumes[vol] = data_frames[j:j + (clip_length - 1) * stride + 1:stride]
            saved_volume = volumes[vol]
            # Create an example protocol buffer
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'volume': _bytes_feature(saved_volume)}
                )
            )
            count += 1
            vol += 1
            if count <= train_size:
                write_1.write(example.SerializeToString())
                if count == train_size:
                    write_1.close()
            else:
                write_2.write(example.SerializeToString())

    write_2.close()
    print('train_num:{}'.format(train_size))
    logger.info('train_num:{}'.format(train_size))
    print('val_num:{}'.format(all_vol_num - train_size))
    logger.info('val_num:{}'.format(all_vol_num - train_size))
    np.savetxt(os.path.join(tfrecord_dir,
                            'num_of_volumes_in_train_stride_{}_validate_split_{}.txt'.format(
                                stride, validate_split)),
               [train_size],
               fmt='%d')
    np.savetxt(os.path.join(tfrecord_dir,
                            'num_of_volumes_in_val_stride_{}_validate_split_{}.txt'.format(
                                stride, validate_split)),
               [all_vol_num - train_size],
               fmt='%d')
    print("finished split for stride_{}".format(stride))
    logger.info("finished split for stride_{}".format(stride))



def preprocess(logger, dataset, clip_length, dataset_path):
    """
    1. frames to npy file
    2. npy into volumes' collection
    :param logger:
    :param dataset:
    :param clip_length:
    :param dataset_path:
    :return:
    """
    logger.info("preprocess for [{}]".format(dataset))

    # to npy
    try:
        for frames_type in ('training_frames', 'testing_frames'):
            frame_path = os.path.join(dataset_path, dataset, frames_type)
            for frames_folder in os.listdir(frame_path):
                npy_file_path = os.path.join(dataset_path, 'npy_dataset', dataset, frames_type,
                         '{}_{}.npy'.format(frames_type, frames_folder))
                assert(os.path.isfile(npy_file_path))
    except AssertionError:
        to_npy(dataset, dataset_path, 'training_frames')
        to_npy(dataset, dataset_path, 'testing_frames')
    except:
        print("unexpected error:", sys.exc_info())

    # only for training frames, split training volumes into train and validate
    try:
        for stride in [1]:
            tfrecord_dir = os.path.join(dataset_path, 'tfrecord_dataset', dataset,
                                        'clip_length_{:02d}'.format(clip_length))
            path = os.path.join(tfrecord_dir,
                                'num_of_volumes_in_training_frames_stride_{}.txt'.format(stride))
            assert os.path.isfile(path)
    except AssertionError:
        volumes_counter(dataset, dataset_path, 'training_frames', clip_length, stride=1)
        # volumes_counter(dataset, dataset_path, 'training_frames', clip_length, stride=2)
        # volumes_counter(dataset, dataset_path, 'training_frames', clip_length, stride=3)
    except:
        print("unexcepted error:", sys.exc_info())

    validate_split = 0.15
    try:
        for stride in [1]:
            desfile_1 = os.path.join(tfrecord_dir,
                                     'train_stride_{}_validate_split_{}.tfrecords'.format(stride,
                                                                                  validate_split))
            desfile_2 = os.path.join(tfrecord_dir,
                                     'val_stride_{}_validate_split_{}.tfrecords'.format(stride,
                                                                                validate_split))
            assert os.path.isfile(desfile_1) and os.path.isfile(desfile_2)
    except AssertionError:
        to_tfrecord_split(dataset, dataset_path, 'training_frames', clip_length, stride=1,
                          validate_split=0.15)
        # to_tfrecord_split(dataset, dataset_path, 'training_frames', clip_length, stride=2, validate_split=0.15)
        # to_tfrecord_split(dataset, dataset_path, 'training_frames', clip_length, stride=3, validate_split=0.15)
    except:
        print("unexcepted error:", sys.exc_info())

    # confuse tfrecord

    print('complete [{}]'.format(dataset))
    logger.info('complete [{}]'.format(dataset))


if __name__ == '__main__':
    clip_length = 5
    # dataset_path = 'test'
    dataset_path = '/home/orli/Blue-HDD/1_final_lab_/Dataset/cgan_data'
    preprocess(logger=logger, dataset='avenue', clip_length=clip_length, dataset_path=dataset_path)
    # preprocess(logger=logger, dataset='ped1', clip_length=clip_length, dataset_path=dataset_path)
    # preprocess(logger=logger, dataset='ped2', clip_length=clip_length, dataset_path=dataset_path)