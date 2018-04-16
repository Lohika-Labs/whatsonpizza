#-*- coding: utf-8 -*-
""" Backend logging module """
import logging

from .common import TITLE

logging.basicConfig()
logger = logging.getLogger(TITLE)  # pylint:disable=invalid-name
logger.setLevel(logging.INFO)
