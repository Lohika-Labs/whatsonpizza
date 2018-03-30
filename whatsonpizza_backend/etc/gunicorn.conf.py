import multiprocessing

# Server socket
bind = "0.0.0.0:6000"
# Number of workers
workers = 1 #multiprocessing.cpu_count() * 2 + 1
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
# Logging
#errorlog = "/tmp/wc_error.log"
loglevel = 'debug'
capture_output = True
