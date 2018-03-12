#!/usr/bin/env python3

import json
import time

from flask import Flask, render_template
from flask_httpauth import HTTPBasicAuth

from classifier import Classifier

app = Flask(__name__)
auth = HTTPBasicAuth()

@auth.get_password
def get_pw(username):
    users = json.loads(open('users.json', 'r').read())
    if username in users:
        return users.get(username)
    return None

@app.route('/favicon.ico')
@auth.login_required
def favicon_page():
    return redirect('/static/img/favicon.ico')

@app.route('/')
@app.route('/index.htm')
@app.route('/index.html')
@auth.login_required
def index_page():
    return render_template('index.html', timestamp=int(time.time()))

@app.route('/image/<image_id>')
@auth.login_required
def serve_image(image_id):
    response =  make_response(None)#classifier.image_data(image_id))
    response.headers['Content-type'] = 'image/jpeg'
    return response

@app.route('/images/')
@app.route('/images')
@auth.login_required
def images_page():
    username = auth.username()
    data = {}
    data['data'] = result
    data['username'] = username
    data['progress'] = {}#classifier.get_progress(username)
    response = make_response(json.dumps(data))
    response.headers['Content-type'] = 'application/json'
    return response


if __name__ == '__main__':
    app.run(port=5000, debug=True)
