#-*- coding: utf-8 -*-

import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
# Number of workers
workers = 4
# multiprocessing.cpu_count()
# The type of workers to use
worker_class = "eventlet"
# Number of pending connections
backlog = 64
# Workers that are silent for more than seconds specified are killed and restarted.
timeout = 60 * 3
# restart every hour if request takes about 30 seconds to complete
max_requests = 128
# A base to use with setproctitle for process naming.
proc_name = 'whatsonpizza_backend'
# A filename to use for the PID file.
pidfile = '/tmp/wb.pid'
# Daemonize the Gunicorn process. Not for Supervisord.
# daemon = True

# Environment variables. Uncomment this line to turn on debugging.
#raw_env = ["WOP_DEBUG=1"]

# Logging
#errorlog = "/tmp/wc_error.log"
loglevel = 'debug'
capture_output = True
