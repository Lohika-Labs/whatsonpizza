#!/usr/bin/env bash

CMD="python3 -m mxnet.tools.im2rec"

${CMD} --list --recursive --num-thread 16 data-train /home/vbartko/projects/wop_data/train/
${CMD} --list --recursive --num-thread 16 data-val /home/vbartko/projects/wop_data/test/
${CMD} --list --recursive --num-thread 16 data-test /home/vbartko/projects/wop_data/orig/
${CMD} --resize 299 --quality 90 --num-thread 16 data-val /home/vbartko/projects/wop_data/test/
${CMD} --resize 299 --quality 90 --num-thread 16 data-train /home/vbartko/projects/wop_data/train/
${CMD} --resize 299 --quality 90 --num-thread 16 data-test /home/vbartko/projects/wop_data/orig/