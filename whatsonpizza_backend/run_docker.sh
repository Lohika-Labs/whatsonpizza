#!/bin/bash

docker build -t wop .
docker run -p 5000:5000 -it wop
