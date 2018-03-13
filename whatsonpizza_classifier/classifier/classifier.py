import json
import os
from base64 import urlsafe_b64encode
from glob import glob
from tempfile import mkstemp

import requests
from PIL import Image

BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
DATASET_DIR = os.path.join(BASEDIR, '..', 'dataset_18k', 'unclassified', '')
TAXONOMY_DIR = os.path.join(BASEDIR, '..', 'taxonomy')
TAXONOMY_FILE = os.path.join(TAXONOMY_DIR, 'pizza_types.json')
TAXONOMY_IMAGES = os.path.join(TAXONOMY_DIR, 'images', '')

class Classifier(object):
    def get_images(self, limit=1):
        image_list = []
        for dirpath, dirnames, fnames in os.walk(DATASET_DIR):
            for fname in fnames:
                if not fname.endswith('.jpg'):
                    continue
                image_list.append(fname)
        return image_list[:limit]

    def get_option_images(self, limit=1000000):
        image_list = []
        for dirpath, dirnames, fnames in os.walk(TAXONOMY_IMAGES):
            for fname in fnames:
                if not fname.endswith('.jpg'):
                    continue
                image_list.append(os.path.join(os.path.split(dirpath)[1], fname))
        print (image_list)
        return image_list

    @staticmethod
    def render_image(filename):
        tmp = mkstemp()[1]
        im = Image.open(filename)
        im.thumbnail((600, 600), Image.ANTIALIAS)
        im.save(tmp + '_thumbnail.jpg', "JPEG")
        with open(tmp + '_thumbnail.jpg', 'rb') as src:
            data = src.read()
        os.remove(tmp + '_thumbnail.jpg')
        return data

    def render_missing_image(self):
        return self.render_image(os.path.join(BASEDIR, 'static', 'img', 'noimg.jpg'))

    def image_data(self, image_id):
        valid_ids = self.get_images(limit=1000000)
        if not image_id in valid_ids:
            return self.render_missing_image()
        filename = os.path.join(DATASET_DIR, image_id)
        return self.render_image(filename)

    def option_image_data(self, image_id):
        valid_ids = self.get_option_images()
        if not image_id in valid_ids:
            print (valid_ids, image_id)
            return self.render_missing_image()
        filename = os.path.join(TAXONOMY_IMAGES, image_id)
        return self.render_image(filename)

    def get_options(self):
        options = []
        with open(TAXONOMY_FILE, 'r') as fh:
            taxonomy = json.loads(fh.read()).get('pizza_types', [])
        for obj_type in taxonomy:
            text = obj_type.get('name', '')
            oid = urlsafe_b64encode(bytes(text, 'utf-8')).decode()
            image_url = glob(os.path.join(TAXONOMY_IMAGES, text, '') + '*.jpg')
            if not image_url:
                image_url = '/option_image/noimg.jpg'
            else:
                image_url = '/option_image/' + text + '/' + os.path.basename(image_url[0])
            options.append({'text': text, 'id': oid, 'image_url': image_url})
        return options
