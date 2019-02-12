#!/bin/bash
echo "hello world!"
# 赋值语句等号两边不能有空格，而字符串比较，等号两边必须有空格
filename='frames'
if [ -d ${filename} ]
then
    rm ${filename} -r
fi
mkdir ${filename}

# get all video file
for file in $(ls *mp4)
do
    #使用变量前加$
    # echo $file
    # 从右往左，非贪婪匹配%右侧通配符的字符串
    name=${file%.*}
    mkdir ${filename}/${name}
    echo 'file created!'
    ffmpeg -i ${file} ${filename}/${name}/%04d.jpg
done
