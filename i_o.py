#!/usr/bin/env

import sys, logging

FORMAT = '%(asctime)s %(filename)-15s %(levelname)-10s: %(message)s'
logging.basicConfig(format=FORMAT)

def getLogger(verbosity=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity)
    return logger
