#!/usr/bin/env bash

CMD="python3 -m mxnet.tools.im2rec"

${CMD} --list --recursive --num-thread 16 data-train data/train/
${CMD} --list --recursive --num-thread 16 data-val data/orig/
${CMD} --resize 250 --quality 90 --num-thread 16 data-val data/orig/
${CMD} --resize 250 --quality 90 --num-thread 16 data-train data/train/
