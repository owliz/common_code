import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os
import scipy.io as scio

"""
def plot_data(data, start_id=0):
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # plot regularity score
    ax.plot(np.arange(start_id, start_id + data.shape[0]), data,
            color='b', linewidth=2.0)
    plt.xlabel('Frame number')
    plt.ylabel('Regularity score')
    plt.ylim(0, 1)
    plt.xlim(1, data.shape[0] + 1)
    plt.legend()
    plt.show()


def plot_min_and_max(data, start_id=0):
    x = np.arange(start_id, start_id + data.shape[0])
    a = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1 # local min+max
    b = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1 # local min
    c = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1 # local max
    # graphical output...
    plt.figure()
    plt.plot(x,data)
    plt.plot(x[b], data[b], "o", label="min")
    # plt.plot(x[c], data[c], "o", label="max")
    plt.legend()
    plt.show()


def find_local_min(x):
    # for local maxima
    maxima_id = argrelextrema(x, np.greater)
    # for local minima
    minima_id = argrelextrema(x, np.less)
    print('maxima_id:{}'.format(maxima_id))
    print('minima_id:{}'.format(minima_id))
"""

def load_groundtruth(dataset, gt_root_dir):
    gt_dir = os.path.join(gt_root_dir, dataset, '{}.mat'.format(dataset))
    assert(os.path.isfile(gt_dir))
    # return gt
    abnormal_events = scio.loadmat(gt_dir, squeeze_me=True)['gt']
    # abnormal_events 三维， [[[1:3],[5:9]], ]
    # 加一维度
    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0],
                                                  abnormal_events.shape[1])
    return abnormal_events


def detected_regions(minima_dir):
    video_nums = len(os.listdir(minima_dir))
    regions = []
    assert video_nums is not None, '[!!!] video_nums is None'
    for idx in range(video_nums):
        minimas = np.loadtxt(os.path.join(minima_dir, 'minima_{:02d}.txt'.format(idx+1)), dtype=int)
        minimas.sort(axis=0)
        # minimas = np.array([1,51,200])
        rows = minimas.shape[0]
        region = [[],[]]
        if rows == 0:
            regions.append(region)
            continue
        start = max(minimas[0]-50, 1)
        end = minimas[0] + 50
        for i in range(1, rows):
            minima = minimas[i]
            if end >= minima:
                end = minima + 50
            else:
                region[0].append(start)
                region[1].append(end)
                start = max(minima-50, 1)
                end = minima + 50
            if i == rows - 1:
                region[0].append(start)
                region[1].append(end)
        region_np = np.array(region) -1
        regions.append(region_np)
    return regions


def event_counter(regularity_score_dir, dataset, gt_root_dir, minima_dir):
    IGNORED_FRAMES = 4
    video_length_list = np.loadtxt(os.path.join(regularity_score_dir, 'video_length_list.txt')).tolist()
    video_nums = len(os.listdir(minima_dir))

    abnormal_events = load_groundtruth(dataset, gt_root_dir)
    assert len(abnormal_events) == video_nums, 'the number of groundTruth does not match inference result'
    # shape  (video_nums, 2, single_video_event_nums)
    regions = detected_regions(minima_dir)
    gt_nums = 0
    detected_nums = 0
    gt_event_counter = 0
    correct_detected = []
    # for avenue, it's 21 (vedio numbers in test file)
    num_video = abnormal_events.shape[0]
    for i in range(num_video):
        video_length = int(video_length_list[i])
        sub_abnormal_events = abnormal_events[i]
        # avenue: abnormal_events[0] 's shape: (2, 5), 5：有五个异常
        # [[  78  392  503  868  932]
        #  [ 120  422  666  910 1101]]
        # 上下对应， e.g., [77, 119]是异常帧
        # 如果缺失一维
        if sub_abnormal_events.ndim == 1:
            # 加一维度
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
        # avenue:
        # (2, 5), num_abnormal=5
        _, num_gt = sub_abnormal_events.shape

        sub_region = regions[i]
        sub_region += IGNORED_FRAMES
        _, num_detected = sub_region.shape

        gt_nums += num_gt
        detected_nums += num_detected

        detected_list = np.zeros((video_length,), dtype=np.int8)
        for j in range(num_detected):
            # final_frame = video_length - 1
            detected_region = [sub_region[0, j], min(sub_region[1, j], video_length-1)]
            detected_list[detected_region[0]:detected_region[1]+1] = 1
        for j in range(num_gt):
            gt_region = [sub_abnormal_events[0, j] - 1, sub_abnormal_events[1, j] - 1]
            over_lapped = np.sum(detected_list[gt_region[0]:gt_region[1]+1] == 1)
            over_lapped_rate = over_lapped / (gt_region[1] - gt_region[0] + 1)
            if over_lapped_rate >= 0.5:
                correct_detected.append(gt_event_counter)
            gt_event_counter += 1

        detected_event_counter = 0
        false_alarm = []
        gt_list = np.zeros((video_length,), dtype=np.int8)
        for j in range(num_gt):
            # final_frame = video_length - 1
            gt_region = [sub_abnormal_events[0, j] - 1, sub_abnormal_events[1, j] - 1]
            gt_list[gt_region[0]:gt_region[1] + 1] = 1
        for j in range(num_detected):
            detected_region = [sub_region[0, j], min(sub_region[1, j], video_length-1)]
            over_lapped = np.sum(gt_list[detected_region[0]:detected_region[1] + 1] == 1)
            over_lapped_rate = over_lapped / (detected_region[1] - detected_region[0] + 1)
            if over_lapped_rate < 0.5:
                false_alarm.append(detected_event_counter)
                detected_event_counter += 1

    return gt_nums, detected_nums, correct_detected, false_alarm


if __name__ == '__main__':
    threshold = 0.1
    dataset = 'ped2'
    #
    # file_name = 'myModel_10/'
    minima_dir = '/home/orli/Blue-HDD/1_final_lab_/exp_data/psnr/minima_{:.1f}/'.format(threshold)

    regularity_score_dir = '/home/orli/Blue-HDD/1_final_lab_/exp_data/psnr/'
    dataset_root_dir = '/home/orli/Blue-HDD/1_final_lab_/Dataset'
    gt_root_dir = dataset_root_dir + '/cgan_data'

    gt_nums, detected_nums, correct_detected, false_alarm = event_counter(regularity_score_dir,
                                                                          dataset, gt_root_dir,
                                                                          minima_dir)
    plot_score(regularity_score, start_id=0)

    plot_min_and_max(regularity_score, start_id=0)
    #
    # find_local_min(regularity_score)