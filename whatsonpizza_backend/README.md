# Intro

Pizza images backend

## Running using Docker

`make docker`

Wait few minutes to boot up (MacBook Pro (i7, Early 2015) - about 7 minutes)
then go to http://localhost:5000 - you're all set.

## Development

WARNING: Python 3.5+


### Installing dependencies

`make deps`

### Running

`make run`

### Debugging (single flask worker)

`./site.py`


### Debugging (gunicorn)

Edit `etc/gunicorn.conf.py` and uncomment `raw_env` line.
