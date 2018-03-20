python2.7 train_multilabel.py \
--epoch 0 \
--model model/resnet-50 \
--batch-size 64 \
--num-classes 52 \
--data-train dataset_18K/train_data.lst \
--image-train dataset_18K/images/ \
--data-val dataset_18K/val_data.lst \
--image-val dataset_18K/images/ \
--num-examples 15173 \
--lr 0.001 \
--num-epoch 15 \
--save-result snapshots/ \
--save-name multilabel-resnet-50 
#\
#--gpus 0,1
