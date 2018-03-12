import json
import os

import requests
from PIL import Image

class Classifier(object):
    def get_image_by_id(self, image_id):
        data = json.loads({})#self.rcon.hget('viewer', image_id))
        data.pop('url')
        return data

    def image_data(self, image_id):
        metadata = json.loads({})#self.rcon.hget('viewer', image_id))
        url = metadata.get('url')
        filename = mkstemp()[1]
        with open(filename, 'w') as fd:
            r = requests.get(url, stream=True)
            for chunk in r.iter_content(1024):
                fd.write(chunk)
        im = Image.open(filename)
        im.thumbnail((600, 600), Image.ANTIALIAS)
        im.save(filename + '_thumbnail.jpg', "JPEG")
        with open(filename + '_thumbnail.jpg', 'r') as src:
            data = src.read()
        os.remove(filename)
        os.remove(filename + '_thumbnail.jpg')
        return data
