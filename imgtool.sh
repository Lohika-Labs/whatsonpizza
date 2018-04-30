#!/usr/bin/env bash

CMD="python3 -m mxnet.tools.im2rec"

${CMD} --list --recursive --num-thread 16 data-train data/train/
${CMD} --list --recursive --num-thread 16 data-val data/test/
${CMD} --resize 256 --quality 90 --num-thread 16 data-val data/test/
${CMD} --resize 256 --quality 90 --num-thread 16 data-train data/train/
