FROM ubuntu:xenial

# Base
ENV APP_PATH /app
ENV LANG C
ENV LANGUAGE C
ENV LC_ALL C
ENV DEBIAN_FRONTEND noninteractive

# Packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils
RUN apt-get install -y python3-dev python3-pip libgtk2.0-dev libhdf5-dev telnet
RUN pip3 install -U pip

# Ports
EXPOSE 5000

# Copy the application over into the container.
ADD . $APP_PATH

# Install  the application's dependencies.
RUN cd $APP_PATH && pwd && ls && make deps
RUN chmod a+x $APP_PATH/entrypoint.sh
WORKDIR $APP_PATH
ENTRYPOINT $APP_PATH/entrypoint.sh
