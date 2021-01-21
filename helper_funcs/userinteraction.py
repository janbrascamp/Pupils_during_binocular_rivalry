#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Jan Brascamp on 2013-03-28.
Copyright (c) 2013 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import subprocess
import re
import math
import numpy
import datetime
import random

from . import filemanipulation

def printFeedbackMessage(message):
	
	print('*** Feedback: '+message+' ***') 