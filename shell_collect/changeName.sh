#!/bin/bash
# 需要用bash命令
 
for dir in $(ls -F | grep '/$')
do
    let i=1;
    for pic in ${dir}*.jpg
    do  
        # echo mv $pic $(printf "%04d" $i).jpg; 
        mv $pic $(printf "%s%04d" ${dir} $i).jpg; 
        let i++; 
    done
done
