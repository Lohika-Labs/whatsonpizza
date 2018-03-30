#!/usr/bin/env bash
mkdir -p categorized_train
for i in categorized/*; do
    c=`basename $i`
    mkdir -p categorized_train/$c
    for j in `ls $i/*.jpg | gshuf | head -n 60`; do
        mv $j categorized_train/$c/
    done
done