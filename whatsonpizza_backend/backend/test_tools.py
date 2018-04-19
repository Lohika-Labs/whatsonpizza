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


IMAGES = glob("./testset/*.jpg")

URL = 'http://localhost:5000/p'


def submit_file(filepath, url=URL):
    """ Submit single image to specified WhatsOnPizza backend """
    response = requests.post(url, data=b64encode(open(filepath, 'rb').read()))
    result = json.loads(response.text)
    return result


def submit_files(glob_path, url=URL):
    """ Submit multiple images specified by glob.glob mask """
    result = []
    for img in glob(glob_path):
        result.append(submit_file(img, url=url))
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=TITLE)  # pylint:disable=invalid-name
    parser.add_argument("-u", "--url", action="store", dest="url")
    group = parser.add_mutually_exclusive_group()  # pylint:disable=invalid-name
    group.add_argument("-l", "--local", action="store_true", dest="local")
    group.add_argument("-w", "--web", action="store_true", dest="web")
    args = ["-h"] if len(sys.argv) == 1 else sys.argv[1:]  # pylint:disable=invalid-name
    result_ = parser.parse_args(args)  # pylint:disable=invalid-name

    if result_.local:
        from .backend import Backend
        backend = Backend()  # pylint:disable=invalid-name
        logger.warning("Running local test...")
        for image in IMAGES:
            logger.warning(backend.tensorflow_analyze_image(image))

        for image in IMAGES:
            logger.warning(backend.mxnet_analyze_image(image))
    elif result_.web:
        url_ = result_.url if result_.url else URL  # pylint:disable=invalid-name
        logger.warning(submit_files('./testset/*.jpg', url=url_))
