FROM ubuntu:trusty

ENV APP_PATH /app
# Base
ENV LANG en_US.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN locale-gen en_US en_US.UTF-8
RUN dpkg-reconfigure locales
RUN apt-get update
RUN apt-get install -y python3-pip python-dev python3-dev libgtk2.0-dev libhdf5-dev
RUN pip3 install --upgrade pip


RUN pip3 install gunicorn setproctitle nose pylint numpy nose-timer requests
# Ports
EXPOSE 80 5000 8080 6000
# Copy the application over into the container.
ADD . $APP_PATH

# Install  the application's dependencies.
RUN cd $APP_PATH && pwd && ls && make deps
RUN chmod a+x $APP_PATH/entrypoint.sh
WORKDIR $APP_PATH
ENTRYPOINT $APP_PATH/entrypoint.sh
