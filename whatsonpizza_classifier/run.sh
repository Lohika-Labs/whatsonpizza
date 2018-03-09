#!/bin/bash

curdir=$(pwd)

cd $(dirname $0)

PYTHONPATH=../ gunicorn -c etc/gunicorn.conf.py whatsonpizza_classifier.site:app

cd ${curdir}
