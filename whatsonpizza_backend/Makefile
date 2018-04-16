.PHONY: run

all: deps

deps:
	@pip3 install -r requirements.txt

run:
	@PYTHONPATH=../ gunicorn -c etc/gunicorn.conf.py whatsonpizza_backend.site:app

test:
	@PYTHONPATH=../ python3 -m whatsonpizza_backend.backend.test_tools -l

test_http:
	@PYTHONPATH=../ python3 -m whatsonpizza_backend.backend.test_tools -w

docker:
	@docker build -t wop .
	@docker run -p 5000:5000 -it wop
