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

def ShellCmd(cmdline,username=-1,serveraddress=-1):
	
	if username==-1: #execute a command in the shell on local computer
		command=cmdline
	else: #execute a command in the shell on the remote computer
		command="ssh "+username+"@"+serveraddress+" "+cmdline
	
	print("executing "+command)
	
	reply = subprocess.Popen(command,
						shell=True,
						stdout=subprocess.PIPE,
						stderr=subprocess.PIPE)
	result = reply.stdout.readlines()
	
	return result
	
def putFileNamesInArray(thepath,username=-1,serveraddress=-1):

	filenames=ShellCmd("ls "+thepath,username,serveraddress)
	filenames=[filename.rstrip('\n') for filename in filenames]	

	return filenames