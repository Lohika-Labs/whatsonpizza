import json
import random
import os
from pathlib import Path
import mxnet as mx
import subprocess


# remove corrupt images
def move_corrupt():
        for filename in os.listdir(DIRimg):
        args = (DIRexe, "-c", DIRimg + filename)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        resp = output.split()[-1]
        
        if resp == '[ERROR]':
            os.rename(DIRimg + filename, '/Users/otkach/Desktop/18K/corrupt/' + filename)


# unique ingridients list
def cats(jsn):
    return set(ing
            for pizza in jsn
            for ing in jsn[pizza]['ingridients']
            )


# one-hot vectors as labels
def cat2vec(ing_lst):
    x = []
    for cat in cats:
        if cat in ing_lst:
            x.append('1.000000')
        else:
            x.append('0.000000')
    return x


def writefile(filename, lst):
    with open(filename + '.lst', 'w') as f:
        for img in lst:
            f.write(
                img['id'] + '\t' + '\t'.join(img['label']) + '\t' + img['path'] + '\n'
            )


def gen_files():
    
    ALL = []

    for pizza in jsn:

        label = cat2vec(jsn[pizza]['ingridients'])

        for img in jsn[pizza]['images']:

            img_path = Path(DIR + img)

            if img_path.exists():

                dict = {}
                dict['id'] = img.replace('.jpg', '')
                dict['label'] = label
                dict['path'] = img
                ALL.append(dict)

    random.shuffle(ALL)

    ntrain = int(len(ALL) * 0.8)
    nval = int(len(ALL) * 0.9)

    writefile('./data_18K/train_data', ALL[:ntrain])
    writefile('./data_18K/val_data', ALL[ntrain:nval])
    writefile('./data_18K/test_data', ALL[nval:])

if __name__ == '__main__':

    JSN_DIR = './dataset_18K/pizza.json'
    DIR = './dataset_18K/images/'
    DIRexe = '/Users/otkach/jpeginfo/jpeginfo'
    DIRimg = './dataset_18K/images/'
    move_corrupt()
    jsn = json.load(open(JSN_DIR))
    cats = cats(jsn)
    gen_files()
