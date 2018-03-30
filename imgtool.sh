#!/usr/bin/env bash
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --num-thread 16 categorized-train categorized_train/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --num-thread 16 categorized-val categorized/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 50 --quality 90 --num-thread 16 categorized-val categorized/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 50 --quality 90 --num-thread 16 categorized-train categorized_train/