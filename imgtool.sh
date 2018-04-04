#!/usr/bin/env bash
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --num-thread 16 categorized-train large_pruned/train/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --list --recursive --train-ratio 0.6 --test-ratio 0.4 --num-thread 16 categorized-val large_pruned/orig/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 50 --quality 90 --num-thread 16 categorized-val large_pruned/orig/
python3 /usr/local/lib/python3.6/site-packages/mxnet/tools/im2rec.py --resize 50 --quality 90 --num-thread 16 categorized-train large_pruned/train/