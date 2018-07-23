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

DEBUG = os.environ.get('WOP_DEBUG', None) is not None

app = Flask(__name__)  # pylint:disable=invalid-name
backend = Backend()  # pylint:disable=invalid-name

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
    tmp = mkstemp(suffix=".jpg")[1]
    with open(tmp, 'wb') as file_handle:
        content = b64decode(request.data)
        file_handle.write(content)
        file_handle.close()
    data = backend.analyze_image(tmp)
    if DEBUG:
        print('Saving analyzed image to `%s`' % tmp)
    else:
        os.remove(tmp)
    response = make_response(json.dumps(data))
    response.headers['Content-type'] = 'application/json'
    print(data)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
