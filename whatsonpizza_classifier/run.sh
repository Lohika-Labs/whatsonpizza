#!/bin/bash

if [ $(python3 -c 'import gunicorn; print(gunicorn.SERVER_SOFTWARE.startswith("gunicorn"))') != "True" ]; then #'
    echo "Has to run on Python3, sorry."
    exit 127
else
    curdir=$(pwd)
    cd $(dirname $0)
    PYTHONPATH=../ gunicorn -c etc/gunicorn.conf.py whatsonpizza_classifier.site:app
    cd ${curdir}
fi
