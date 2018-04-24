#!/usr/bin/env bash
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --num-thread 16 data-train data/train/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --num-thread 16 data-val data/orig/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 250 --quality 90 --num-thread 16 data-val data/orig/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 250 --quality 90 --num-thread 16 data-train data/train/
