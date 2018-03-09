#!/usr/bin/env python3

import json
import time

from flask import Flask, render_template
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()

@auth.get_password
def get_pw(username):
    users = json.loads(open('users.json', 'r').read())
    if username in users:
        return users.get(username)
    return None

@app.route('/')
@app.route('/index.htm')
@app.route('/index.html')
@auth.login_required
def index_page():
    return render_template('index.html', timestamp=int(time.time()))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
