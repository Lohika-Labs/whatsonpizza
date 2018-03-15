#!/usr/bin/env python3

import json
import time

from flask import (Flask,
                   make_response,
                   redirect,
                   render_template,
                   request
                  )

from flask_httpauth import HTTPBasicAuth

from classifier import Classifier

app = Flask(__name__)
auth = HTTPBasicAuth()
classifier = Classifier()

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
    return render_template('index.html', timestamp=int(time.time()), options=classifier.get_options())

@app.route('/image/<image_id>')
@auth.login_required
def serve_image(image_id):
    response =  make_response(classifier.image_data(image_id))
    response.headers['Content-type'] = 'image/jpeg'
    return response


@app.route('/option_image/<path:path>')
@auth.login_required
def serve_option_image(path):
    response =  make_response(classifier.option_image_data(path))
    response.headers['Content-type'] = 'image/jpeg'
    return response


@app.route('/images/')
@app.route('/images')
@auth.login_required
def images_page():
    username = auth.username()
    data = {}
    data['data'] = classifier.get_images()
    data['username'] = username
    data['progress'] = {}#classifier.get_progress(username)
    response = make_response(json.dumps(data))
    response.headers['Content-type'] = 'application/json'
    return response

@app.route('/p/', methods=['POST'])
@app.route('/p', methods=['POST'])
@auth.login_required
def p_page():
    payload = request.values.to_dict()
    classifier.classify_image(payload)
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
