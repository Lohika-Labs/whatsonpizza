#!/usr/bin/env python3

import json
import time

from base64 import b64decode
from tempfile import mkstemp

from flask import (Flask,
                   make_response,
                   redirect,
                   render_template,
                   request
                  )

from backend import Backend

app = Flask(__name__)
backend = Backend()


@app.route('/favicon.ico')
def favicon_page():
    return redirect('/static/img/favicon.ico')

@app.route('/')
@app.route('/index.htm')
@app.route('/index.html')
def index_page():
    return 'Nothing here'


@app.route('/p/', methods=['POST'])
@app.route('/p', methods=['POST'])
def p_page():
    data = {"status": "ok", "status_message": "", "mxnet": [{
                                                                   "value": 0.75,
                                                                   "name": "quattro formaggi"}, {"value": 0.98, "name": "margherita"}],

                                                      "tensorflow": [{"value": 0.75, "name": "quattro stagioni"}]

               } 
    tmp = mkstemp()[1]
    with open(tmp, 'wb') as fh:
        content = b64decode(request.data)
        fh.write(content)
        print (tmp)
    response = make_response(json.dumps(data))
    response.headers['Content-type'] = 'application/json'
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
