#!/usr/bin/env python3

# std
import json
import sys

# std from
from base64 import b64encode
from glob import glob

# 3rd party
import requests

URL='http://localhost:5000/p'

def submit_file(filepath, url=URL):
    r = requests.post(url, data=b64encode(open(filepath, 'rb').read()))
    result = json.loads(r.text)
    return result

def submit_files(glob_path, url=URL):
    result = []
    for img in glob(glob_path):
        result.append(submit_file(img, url=url))
    return result

if __name__ == '__main__':
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = URL
    print (submit_files('./testset/*.jpg', url=url))
