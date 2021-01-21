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
#from scipy.stats import gamma
from scipy.stats import lognorm
import datetime
import random

from IPython import embed as shell
	
def readDelimitedIntoArray(myPath,delimiter):
	
	try:
		f = open(myPath, 'r')
		table = [[float(element) for element in row.strip().split(delimiter)] for row in f]
		
	except ValueError:
		f.close()
		f = open(myPath, 'r')
		table = [[element for element in row.strip().split(delimiter)] for row in f]
		
	f.close()
	
	return table