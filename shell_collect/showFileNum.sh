#!/bin/bash
for dir in $(ls -F | grep '/$')
do
    #echo "file:"${dir}
    ls -lR ${dir}| grep "^-" | wc -l
done
