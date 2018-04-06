#!/usr/bin/env python3

# std
import json

# std from
from base64 import b64encode
from glob import glob

# 3rd party
import requests

URL='http://localhost:5000/p'

def submit_file(filepath):
    r = requests.post(URL, data=b64encode(open(filepath, 'rb').read()))
    result = json.loads(r.text)
    return result

def submit_files(glob_path):
    result = []
    for img in glob(glob_path):
        result.append(submit_file(img))
    return result

if __name__ == '__main__':
    print (submit_files('./testset/*.jpg'))