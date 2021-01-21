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
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import time

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
backupSubFolder='_allInfDictListBackups'
makeBackup=False
miscStuffSubFolder='other'
figuresSubFolder='figures'
behavioralSubFolder='behavioral'
eyeSubFolder='eye'
beforeMergingFolder='before_merging'
timecourseDataSubFolder='filtered'
regressorsSubFolder='pupilRegressors'
GLMoutcomeSubFolder='GLMoutcomes'

excludedObs=['PC','MH']

# print('TEMP!')
# time.sleep(60.*60.*4.)

timingMatchToleranceS=1. #in matching examinedEvents to RefEvents, we're looking in the spot that's just before or after the RefEvent as indicated by the decoPlot, and finding the examinedEvent closest to that spot. How far away (before or after, symmetrically) is the examinedEvent allowed to be from that spot to still be counted a match?

blinkDistanceAllowedS=1.5	#when separating switch regressors into those with and without a blink nearby, what counts as 'nearby'?

allInfoDictListFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_allInfoDictList' in element]
uniqueObservers=[fileName.split('_')[0] for fileName in allInfoDictListFilesPresent]
uniqueObservers=[thisObs for thisObs in uniqueObservers if not thisObs in excludedObs]
#uniqueObservers=uniqueObservers[:2]	#for short debugging runs

allDictLists=[]
for oneObs in uniqueObservers:
	with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', "r") as f:
		oneDictList=pickle.load(f)
	if makeBackup:
		print('backing up '+oneObs+'_allInfoDictList because going to write new one')
		commandline.ShellCmd('mv '+myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList '+myPath+miscStuffSubFolder+'/'+backupSubFolder+'/'+oneObs+'_allInfoDictList')
	allDictLists=allDictLists+[oneDictList]

#--------------------------------------------------------------
#separate inferred switches for passive replay and active rivalry into ones that are and are not instantaneous, based on the physical and reported switches, respectively, that they match

sessionTypes=['Passive replay','Active rivalry']				
refEventKindStrings=['physicalSwitches','reportedSwitches']		#the ones whose transition durations are known
refEventDurationStrings=['physicalSwitchDurations','reportedSwitchDurations']	#where those durations are stored
examinedEventKindStrings=['inferredSwitches','inferredSwitches']	#the ones that are being sorted into instantaneous or not (or unknown), depending on closest of kind 'refEventKindString'
relevantTimeShiftStrings=['passiveReplayPhysicalToInferredShift','rivalryReportedToInferredShift']

for thisVersionIndex in range(len(sessionTypes)):
	
	sessionType=sessionTypes[thisVersionIndex]				
	refEventKindString=refEventKindStrings[thisVersionIndex]
	refEventDurationString=refEventDurationStrings[thisVersionIndex]
	examinedEventKindString=examinedEventKindStrings[thisVersionIndex]
	relevantTimeShiftString=relevantTimeShiftStrings[thisVersionIndex]

	for obsIndex,oneObs in enumerate(uniqueObservers):

		oneDictList=allDictLists[obsIndex]

		thisSessionIndex=[index for index,element in enumerate(oneDictList) if element['sessionKind']==sessionType][0]

		examinedEventList=oneDictList[thisSessionIndex][examinedEventKindString]
		refEventList=oneDictList[thisSessionIndex][refEventKindString]
		refEventDurationList=oneDictList[thisSessionIndex][refEventDurationString]

		decoPeakInS=oneDictList[thisSessionIndex][relevantTimeShiftString]	#how much to shift ref event times to align them as closely as possible with examined event kind

		examinedEvents0duration=[]
		examinedEventsNon0duration=[]

		for oneExaminedEvent in examinedEventList:

			closestRefEventIndex=numpy.argmin([abs(oneExaminedEvent-decoPeakInS-thisRefEvent) for thisRefEvent in refEventList])

			if abs(oneExaminedEvent-decoPeakInS-refEventList[closestRefEventIndex])<timingMatchToleranceS:

				if refEventDurationList[closestRefEventIndex]==0.:
					examinedEvents0duration=examinedEvents0duration+[oneExaminedEvent]
				else:
					examinedEventsNon0duration=examinedEventsNon0duration+[oneExaminedEvent]

			else:	#if you can't reliably match this examined (inferred) event to a reference (physical) event then treat it as non-0

				examinedEventsNon0duration=examinedEventsNon0duration+[oneExaminedEvent]

		oneDictList[thisSessionIndex][examinedEventKindString+'0duration']=numpy.array(examinedEvents0duration)
		oneDictList[thisSessionIndex][examinedEventKindString+'Non0duration']=numpy.array(examinedEventsNon0duration)
	
		allDictLists[obsIndex]=oneDictList

	# with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', 'w') as f:
	#     pickle.dump(oneDictList, f)

#--------------------------------------------------------------
#separate inferred rivalry switches for passive rivalry into ones that do and ones that don't have a motion probe in the immediately following perceptual dominance period. 
sessionType='Passive rivalry'				
examinedEventKindString='inferredSwitches'

for obsIndex,oneObs in enumerate(uniqueObservers):

	oneDictList=allDictLists[obsIndex]

	thisSessionIndex=[index for index,element in enumerate(oneDictList) if element['sessionKind']==sessionType][0]

	examinedEventList=oneDictList[thisSessionIndex][examinedEventKindString]
	trialEndEvents=oneDictList[thisSessionIndex]['trialEnds']
	probeEventsAll=oneDictList[thisSessionIndex]['shownProbes']

	examinedEventsWithProbe=[]
	examinedEventsWithoutProbe=[]

	for examinedEventIndex,oneExaminedEvent in enumerate(examinedEventList[:-1]):

		followingEvent=examinedEventList[examinedEventIndex+1]
		anyTrialEndsBetween=[element for element in trialEndEvents if element>oneExaminedEvent and element<followingEvent]
		if len(anyTrialEndsBetween)==0:
			anyProbesBetween=[element for element in probeEventsAll if element>oneExaminedEvent and element<followingEvent]
			if len(anyProbesBetween)==0:
				examinedEventsWithoutProbe=examinedEventsWithoutProbe+[oneExaminedEvent]
			else:
				examinedEventsWithProbe=examinedEventsWithProbe+[oneExaminedEvent]
		else:
			examinedEventsWithProbe=examinedEventsWithProbe+[oneExaminedEvent]	#call it 'with probe' if it's the last switch of a trial, simply because that type of switch won't be analyzed in the subsequent percept duration covariate analysis anyway
	
	examinedEventsWithProbe=examinedEventsWithProbe+[examinedEventList[-1]]	#very last switch of experiment also goes to 'with probe' for same reason
	
	oneDictList[thisSessionIndex][examinedEventKindString+'NoProbeInSubsequentPerc']=numpy.array(examinedEventsWithoutProbe)
	oneDictList[thisSessionIndex][examinedEventKindString+'ProbeInSubsequentPerc']=numpy.array(examinedEventsWithProbe)

	allDictLists[obsIndex]=oneDictList
	# with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', 'w') as f:
	#     pickle.dump(oneDictList, f)

#--------------------------------------------------------------		
#make empty lists with these two things just computed, but now for the other conditions, not to trip up the '1d_...' code.
sessionTypes=['Passive rivalry','Active replay']
examinedEventKindString='inferredSwitches'	#the ones that are being sorted into instantaneous or not (or unknown), depending on closest of kind 'refEventKindString'

for obsIndex,oneObs in enumerate(uniqueObservers):

	oneDictList=allDictLists[obsIndex]
		
	for sessionType in sessionTypes:
		thisSessionIndex=[index for index,element in enumerate(oneDictList) if element['sessionKind']==sessionType][0]
		
		oneDictList[thisSessionIndex][examinedEventKindString+'0duration']=numpy.array([])
		oneDictList[thisSessionIndex][examinedEventKindString+'Non0duration']=numpy.array([])
		
	allDictLists[obsIndex]=oneDictList

sessionTypes=['Passive replay','Active rivalry','Active replay']
examinedEventKindString='inferredSwitches'	#the ones that are being sorted into whether there's a probe in the subsequent percept

for obsIndex,oneObs in enumerate(uniqueObservers):

	oneDictList=allDictLists[obsIndex]
		
	for sessionType in sessionTypes:
		thisSessionIndex=[index for index,element in enumerate(oneDictList) if element['sessionKind']==sessionType][0]
		
		oneDictList[thisSessionIndex][examinedEventKindString+'NoProbeInSubsequentPerc']=numpy.array([])
		oneDictList[thisSessionIndex][examinedEventKindString+'ProbeInSubsequentPerc']=numpy.array([])
	
	allDictLists[obsIndex]=oneDictList
	#with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', 'w') as f:
	#    pickle.dump(oneDictList, f)

#-------------------------------------------------------------------		
#separate inferred rivalry switches into ones that do and ones that don't have a blink closeby. 
sessionTypes=['Passive rivalry','Active rivalry','Passive replay','Active replay']
examinedEventKindString='inferredSwitches'

for obsIndex,oneObs in enumerate(uniqueObservers):
	
	oneDictList=allDictLists[obsIndex]

	for sessionType in sessionTypes:
		
		thisSessionIndex=[index for index,element in enumerate(oneDictList) if element['sessionKind']==sessionType][0]

		examinedEventList=oneDictList[thisSessionIndex][examinedEventKindString]
		blinkEvents=oneDictList[thisSessionIndex]['blinks']

		examinedEventsWithBlink=[]
		examinedEventsWithoutBlink=[]
		
		for examinedEventIndex,oneExaminedEvent in enumerate(examinedEventList):
			anyBlinksCloseby=[element for element in blinkEvents if element>(oneExaminedEvent-blinkDistanceAllowedS) and element<(oneExaminedEvent+blinkDistanceAllowedS)]
			if len(anyBlinksCloseby)==0:
				examinedEventsWithoutBlink=examinedEventsWithoutBlink+[oneExaminedEvent]
			else:
				examinedEventsWithBlink=examinedEventsWithBlink+[oneExaminedEvent]

		oneDictList[thisSessionIndex][examinedEventKindString+'NoBlink']=numpy.array(examinedEventsWithoutBlink)
		oneDictList[thisSessionIndex][examinedEventKindString+'Blink']=numpy.array(examinedEventsWithBlink)

	allDictLists[obsIndex]=oneDictList	#not strictly necessary as long as you're pickle.dumping here. But becomes necessary if that part gets commented out at some point, as above.
	with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', 'w') as f:
		pickle.dump(oneDictList, f)
		
		
