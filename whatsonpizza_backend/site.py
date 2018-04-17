#!/usr/bin/env python3
#-*- coding: utf-8 -*-
""" Flask application """

import json
import os

from base64 import b64decode
from tempfile import mkstemp

from flask import (Flask,
                   make_response,
                   redirect,
                   request
                  )

from backend.backend import Backend
from backend.detection import PizzaDetectorWrapper

app = Flask(__name__)  # pylint:disable=invalid-name
backend = Backend()  # pylint:disable=invalid-name
pizza_detector = PizzaDetectorWrapper()

@app.route('/favicon.ico')
def favicon_page():
    """ Return favicon """
    return redirect('/static/img/favicon.ico')

@app.route('/')
@app.route('/index.htm')
@app.route('/index.html')
def index_page():
    """ Empty index page """
    return 'Nothing here'


@app.route('/p/', methods=['POST'])
@app.route('/p', methods=['POST'])
def p_page():
    """ API endpoint """
    data = {"status": "ok", "status_message": ""}
    tmp = mkstemp(suffix=".jpg")[1]
    with open(tmp, 'wb') as file_handle:
        content = b64decode(request.data)
        file_handle.write(content)
        file_handle.close()
    if not pizza_detector.detect_pizza(tmp):
        data = {"status": "error", "status_message": "Not a pizza"}
    else:
        data['mxnet'] = backend.mxnet_analyze_image(tmp)
        data['tensorflow'] = backend.tensorflow_analyze_image(tmp)
    os.remove(tmp)
    response = make_response(json.dumps(data))
    response.headers['Content-type'] = 'application/json'
    print(data)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
