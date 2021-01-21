#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Jan Brascamp on 2017-06-29.
Copyright (c) 2017 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import numpy
import re
import scipy as sp
import scipy.signal
import pickle
from scipy.stats import ttest_1samp
import scipy.stats

from IPython import embed as shell

import matplotlib
import matplotlib.pyplot as pl

import helper_funcs.commandline	as commandline
import helper_funcs.analysis as analysis
import helper_funcs.userinteraction as userinteraction
import helper_funcs.filemanipulation as filemanipulation

# import seaborn as sn
# sn.set(style="ticks")

#myPath='/Users/janbrascamp/Documents/Experiments/pupil_switch/fall_18_dontcorrectforshortening/data/'
myPath='/Users/janbrascamp/Dropbox/__FS20/Experiments/fall_18_dontcorrectforshortening/data/'
miscStuffSubFolder='other'
figuresSubFolder='figures'
behavioralSubFolder='behavioral'
eyeSubFolder='eye'
beforeMergingFolder='before_merging'
timecourseDataSubFolder='filtered'
regressorsSubFolder='pupilRegressors'
GLMoutcomeSubFolder='GLMoutcomes'

obsLeftOut=['PC','MH']

analysis.plotReportedSwitchInfo(myPath+regressorsSubFolder+'/',myPath+figuresSubFolder+'/',myPath+behavioralSubFolder+'/',obsLeftOut)

switchRateComparisonFiles=[element for element in os.listdir(myPath+miscStuffSubFolder) if '_switch_rate_comparison_data' in element]

allInferredRates=[list(numpy.load(myPath+miscStuffSubFolder+'/'+thisFile)) for thisFile in switchRateComparisonFiles]	#oneObs,sessionKind,inferredPerSecond,reportedPerSecond,physicalPerSecond

#allInferredRates=[element for element in allInferredRates if element[0] in ['AB','AD','AG','AF','CC','SG','AM','AO','JE','JB','KC','SF','MB','MY','NB','MH','KM','ST','PC','MW','PA','LS']]

allInferredRates.sort()	#this makes sure all files with corresponding participant names are grouped together, which is important for plots (below) that combine across conditions.

#x,y,title
interestingCombis=[[[('0','0'),3],[('0','0'),2],"Active rivalry inferred vs reported"],[[('0','0'),2],[('0','1'),2],"Passive rivalry inferred vs active rivalry inferred"],[[('1','0'),3],[('1','0'),4],"Active on-screen physical vs reported"],[[('1','0'),3],[('1','0'),2],"Active on-screen inferred vs reported"],[[('1','0'),4],[('1','0'),2],"Active on-screen inferred vs physical"],[[('1','0'),2],[('1','1'),2],"Passive on-screen inferred vs active on-screen inferred"],[[('1','1'),4],[('1','1'),2],"Passive on-screen inferred vs physical"]]
axisLabelWords=['poop','pee','inferred','reported','physical']
axisLabelBase='Switches per second, '
fig = pl.figure(figsize = (20,20))
for imageIndex,interestingCombi in enumerate(interestingCombis):

	s = fig.add_subplot(3,3,imageIndex+1)

	xData=[element[interestingCombi[0][1]] for element in allInferredRates if element[1]==interestingCombi[0][0]]
	yData=[element[interestingCombi[1][1]] for element in allInferredRates if element[1]==interestingCombi[1][0]]

	xAxisText=axisLabelBase+axisLabelWords[interestingCombi[0][1]]
	yAxisText=axisLabelBase+axisLabelWords[interestingCombi[1][1]]
	
	textData=[element[0] for element in allInferredRates if element[1]==interestingCombi[1][0]]
	
	dropIndices=[index for index,element in enumerate(textData) if element in obsLeftOut]

	xDataPlotted=[element for index,element in enumerate(xData) if not index in dropIndices]
	yDataPlotted=[element for index,element in enumerate(yData) if not index in dropIndices]
	textDataUsed=[element for index,element in enumerate(textData) if not index in dropIndices]
	
	slope,offset = numpy.polyfit(xDataPlotted, yDataPlotted, 1)

	pl.scatter(xDataPlotted,yDataPlotted)
	pl.plot([0,1.1*max(xDataPlotted+yDataPlotted)],[0,1.1*max(xDataPlotted+yDataPlotted)])
	pl.plot(numpy.linspace(0,1.1*max(xDataPlotted+yDataPlotted),10),[offset+slope*xVal for xVal in numpy.linspace(0,1.1*max(xDataPlotted+yDataPlotted),10)])
	[pl.text(xDataPlotted[thisIndex], yDataPlotted[thisIndex]-.04, textDataUsed[thisIndex], fontsize=8) for thisIndex in range(len(xDataPlotted))]
	
	pearsonRandP=scipy.stats.pearsonr(xDataPlotted, yDataPlotted)
	s.set_title(interestingCombi[-1]+'\nr='+str(round(pearsonRandP[0]*10000.)/10000.)+'\np='+str(round(pearsonRandP[1]*10000.)/10000.))

	s.set_xlim(xmin =0, xmax = 1.1*max(xDataPlotted+yDataPlotted))
	s.set_ylim(ymin =0., ymax = 1.1*max(xDataPlotted+yDataPlotted))
	
	s.set_xlabel(xAxisText)
	s.set_ylabel(yAxisText)
	

pl.savefig(myPath+figuresSubFolder+'/switch_rate_correlations.pdf')
pl.close()