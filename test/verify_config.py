#!/usr/bin/env python3
import os

KEEP_FILES = os.environ.get("METIQ_KEEP_FILES", False)
DEBUG = int(os.environ.get("METIQ_DEBUG", 1))
PREC = 0.01  # sec
