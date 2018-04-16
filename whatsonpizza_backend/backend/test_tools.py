#-*- coding: utf-8 -*-
""" Backend tests """

# std
import json
import sys

# std from
from argparse import ArgumentParser
from base64 import b64encode
from glob import glob

# 3rd party
import requests

# local
from .common import TITLE
from .logger import logger


images = glob("./testset/*.jpg")

URL = 'http://localhost:5000/p'


def submit_file(filepath, url=URL):
    r = requests.post(url, data=b64encode(open(filepath, 'rb').read()))
    result = json.loads(r.text)
    return result


def submit_files(glob_path, url=URL):
    result = []
    for img in glob(glob_path):
        result.append(submit_file(img, url=url))
    return result






if __name__ == "__main__":
    parser = ArgumentParser(description=TITLE)
    parser.add_argument("-u", "--url", action="store", dest="url")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", action="store_true", dest="local")
    group.add_argument("-w", "--web", action="store_true", dest="web")
    args = ["-h"] if len(sys.argv) == 1 else sys.argv[1:]
    result = parser.parse_args(args)

    if result.local:
        from .backend import Backend
        b = Backend()
        logger.warning("Running local test...")
        for img in images:
            logger.warning(b.tensorflow_analyze_image(img))

        for img in images:
            logger.warning(b.mxnet_analyze_image(img))
    elif result.web:
        url = result.url if result.url else URL
        logger.warning(submit_files('./testset/*.jpg', url=url))




