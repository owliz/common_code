import os
from skimage import color
from skimage.transform import resize
from skimage.io import imread, imsave

size = (227, 227)
basedir = '.'

def resizeImg(dataset):
    for dir_nm in os.listdir(os.path.join(basedir,dataset)):
        dir_path = os.path.join(basedir,dataset, dir_nm)
        print('---> '+dir_path)
        for file_nm in os.listdir(dir_path):
            if file_nm.lower().endswith(('jpg','png')):
                img = imread(os.path.join(dir_path, file_nm)) # 依次读取rgb图片
                # img=color.rgb2gray(img) #将rgb图片转换成灰度图 
                # model's choice is same as numpy.pad.xx
                img = resize(img, size, mode='reflect')
                imsave(os.path.join(dir_path, file_nm), img)     # save frame as JPG file
    print('success')

resizeImg('avenue/training/frames')
resizeImg('ped1/training/frames')