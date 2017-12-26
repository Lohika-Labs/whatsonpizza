import json
import random
from PIL import Image
import os

JSN_DIR = 'dataset_18K/pizza.json'
IMG_DIR = 'dataset_18K/images/'

# remove corrupt images
def move_corrupt(DIR):
    for filename in os.listdir(DIR):
        try:
            img = Image.open(DIR + filename)
            img.verify()
        except:
            os.rename(DIR + filename, './corrupt/' + filename)


# unique ingridients list
def cats(jsn):
    return set(ing
            for pizza in jsn
            for ing in jsn[pizza]['ingridients']
            )

def save_cats(cats):
    with open('./data_18K/cats.txt', 'w') as f:
        for cat in cats:
            f.write(cat + '\n')

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
    pizza_ids = []
    for pizza in jsn:
        pizza_ids.append(pizza)
        for img in jsn[pizza]['images']:

            try:
                im = Image.open(IMG_DIR + img)
                im.verify()

                dict = {}
                label = cat2vec(jsn[pizza]['ingridients'])
                dict['id'] = img.replace('.jpg', '')
                dict['label'] = label
                dict['path'] = img
                ALL.append(dict)

            except: continue

    random.shuffle(ALL)

    ntrain = int(len(ALL) * 0.8)
    nval = int(len(ALL) * 0.9)

    writefile('./data_18K/train_data', ALL[:ntrain])
    writefile('./data_18K/val_data', ALL[ntrain:nval])
    writefile('./data_18K/test_data', ALL[nval:])


move_corrupt(IMG_DIR)
jsn = json.load(open(JSN_DIR))
cats = list(cats(jsn))
save_cats(cats)
gen_files()
