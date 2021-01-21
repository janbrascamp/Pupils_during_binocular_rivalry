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
#import scipy as sp
import pickle
from scipy.stats import ttest_1samp

from IPython import embed as shell

import matplotlib
import matplotlib.pyplot as pl

import helper_funcs.commandline	as commandline
import helper_funcs.analysis as analysis
import helper_funcs.userinteraction as userinteraction
import helper_funcs.filemanipulation as filemanipulation

#import FIRDeconvolution
from scipy import optimize
import scipy
import scipy.signal
import nideconv

import time

#myPath='/Users/janbrascamp/Documents/Experiments/pupil_switch/fall_18_dontcorrectforshortening/data/'
myPath='/Users/janbrascamp/Dropbox/__FS20/Experiments/fall_18_dontcorrectforshortening/data/'
miscStuffSubFolder='other'
figuresSubFolder='figures'
behavioralSubFolder='behavioral'
eyeSubFolder='eye'
beforeMergingFolder='before_merging'

excludedObs=['PC','MH']

trialDurS=60.

basicSampleRate=1000	#Hz, of the recorded data in the pupil file

#print('TEMP!')
#time.sleep(60.*60.*5.)

downSampleFactors=[10,10]	#downsample in multiple steps if total downSampleFactor>13. Recommended for decimate function
downSampleFactor=downSampleFactors[0]
for additionalFactor in downSampleFactors[1:]:
	downSampleFactor=downSampleFactor*additionalFactor
	
newSampleRate=basicSampleRate/downSampleFactor

downsampledSamplesPerTrial=int(trialDurS*float(newSampleRate))

decoInterval=[-3.5,6.5]	#for pupil data; not behavioral data. was [-3.5,5.]
decoIntervalSacc=[-.5,4.5]	#was [-1.5,6.5]
decoIntervalBlink=[-.5,7.5]	#was [-1.5,6.5]
print('make sure deco intervals match those used in 1_rivalry_task_GT0920 when creating these allInfoDictLists!')

# baselineInterval=[-3.5,-3.]
# subtractBaselineForPlots=False
# ridgeForrester=False

numBasisFunctionsIncOffset=int((decoInterval[1]-decoInterval[0])*2.+1.)	#used to be 25. Now adaptively whatever is 1 Hz
numBasisFunctionsIncOffsetSacc=int((decoIntervalSacc[1]-decoIntervalSacc[0])*2.+1.)
numBasisFunctionsIncOffsetBlink=int((decoIntervalBlink[1]-decoIntervalBlink[0])*2.+1.)

minNumEvs=5 		#if a regressor has fewer than this number of events, then don't try.

sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
timeSeriesString='paddedPupilSamplesSaccBlinksNotRemoved'
regressorStrings=['trialStarts','saccades','blinks',
'reportedSwitches','reportedSwitches0duration','reportedSwitchesNon0duration',
'physicalSwitches','physicalSwitches0duration','physicalSwitchesNon0duration',
'inferredSwitches','inferredSwitches0duration','inferredSwitchesNon0duration',
'inferredSwitchesNoBlink','inferredSwitchesBlink',
'shownProbes','probeReports','unreportedProbes',
'reportedSwitchStarts','reportedSwitchEnds',
'physicalSwitchStarts','physicalSwitchEnds']

shiftStrings=['rivalryInferredToReportedShift',
'rivalryReportedToInferredShift',
'replayReportedToInferredShift',
'replayInferredToReportedShift',
'replayPhysicalToReportedShift',
'replayPhysicalToInferredShift']

joinSets=[[['reportedSwitches0duration_Any','reportedSwitchStarts_Any','reportedSwitchEnds_Any','probeReports_Any'],'keyPress_Any'],
[['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry'],'inferredRivalrySwitch_Any'],
[['inferredSwitches_Active replay','inferredSwitches_Passive replay'],'inferredReplaySwitch_Any']]

# allInfDctArrayFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_allInfDctArray' in element]
# uniqueObservers=[fileName.split('_')[0] for fileName in allInfDctArrayFilesPresent]
# 
# allInfoDictLists=[]
# for oneObs in uniqueObservers:
# 
# 	allInfDctArrayThisObs=numpy.load(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfDctArray.npy', allow_pickle=True)
# 	allInfoDictListThisObs=list(allInfDctArrayThisObs)
# 	
# 	# with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', "rb") as f:
# 	# 	allInfoDictListThisObs=pickle.load(f)
# 	
# 	allInfoDictLists=allInfoDictLists+[allInfoDictListThisObs]

allInfoDictListFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_allInfoDictList' in element]
uniqueObservers=[fileName.split('_')[0] for fileName in allInfoDictListFilesPresent]
uniqueObservers=[element for element in uniqueObservers if not element in excludedObs]

# uniqueObservers=uniqueObservers[:2]
# print('kutting de kut')

allInfoDictLists=[]
for oneObs in uniqueObservers:

	with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', "r") as f:
		allInfoDictListThisObs=pickle.load(f)
	
	allInfoDictLists=allInfoDictLists+[allInfoDictListThisObs]
	
concatenatedDictList=[]
for oneDictList in allInfoDictLists:
	
	totalOffset=0.	#how much do the regressors have to be shifted by (in seconds) to align with the concatenated data
	
	concatenatedDict={'obs':oneDictList[0]['obs']}
	concatenatedData=numpy.empty(0)
	
	for oneSessionKind in sessionKinds:
		
		thisDict=[element for element in oneDictList if element['sessionKind']==oneSessionKind][0]
		
		concatenatedData=numpy.append(concatenatedData,thisDict[timeSeriesString])
		
		for oneRegressorString in regressorStrings:
			
			if oneRegressorString in thisDict.keys():
				
				concatenatedDict[oneRegressorString+'_'+oneSessionKind]=thisDict[oneRegressorString]+totalOffset
		
		for oneShiftString in shiftStrings:
			
			if oneShiftString in thisDict.keys():
				
				concatenatedDict[oneShiftString]=thisDict[oneShiftString]
		
		totalOffset=totalOffset+float(len(thisDict[timeSeriesString]))/float(basicSampleRate)
		
	#---decimate the signal
	downsampledSignal=concatenatedData[:]
	
	for oneFactor in downSampleFactors:
		downsampledSignal=scipy.signal.decimate(downsampledSignal,oneFactor)
	
	concatenatedDict['pupilData']=downsampledSignal	#concatenatedDict gets downsampled pupil data
	#----------------------
	
	derivativeOfSignal=numpy.diff(concatenatedData)	#shifted by 1/2 sample interval at the original basicSampleRate. Not doing anything with that information because it's 1/2 of a ms usually. Printing a warning, though.
		
	#---decimate the derivative of the signal
	downsampledSignal=derivativeOfSignal[:]
	
	for oneFactor in downSampleFactors:
		downsampledSignal=scipy.signal.decimate(downsampledSignal,oneFactor)
	
	concatenatedDict['pupilDataDeriv']=downsampledSignal	#concatenatedDict gets downsampled derivative data
	print('Warning: if working with derivatives, the pupil signal has been shifted by '+str(0.5/float(basicSampleRate))+' s.')
	#----------------------	

	for oneRegressorString in regressorStrings:	#join regressors of a given kind across conditions
		
		jointRegressorAcrossConditions=numpy.empty(0)
		
		for oneSessionKind in sessionKinds:
			
			if oneRegressorString+'_'+oneSessionKind in concatenatedDict.keys():
				
				jointRegressorAcrossConditions=numpy.append(jointRegressorAcrossConditions,concatenatedDict[oneRegressorString+'_'+oneSessionKind])
				
		jointRegressorAcrossConditions.sort()
		concatenatedDict[oneRegressorString+'_Any']=jointRegressorAcrossConditions

	for oneJoinSet in joinSets:	#join regressors of various kinds into a single regressor (e.g. all key presses)
		
		sourceData=numpy.empty(0)
		
		for oneSourceRegressor in oneJoinSet[0]:
			sourceData=numpy.append(sourceData,concatenatedDict[oneSourceRegressor])
		sourceData.sort()
		
		concatenatedDict[oneJoinSet[1]]=sourceData
		
	concatenatedDictList=concatenatedDictList+[concatenatedDict]
	
del allInfoDictLists		#clear up all that memory

#----------------
#define GLMS

plotDictList=[]

#Figures for paper:

plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_OKN'
regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
plottedRegressorIndices=[[0,1,2,3,5,7,8]]
timeShifts=[[0,0,0,0,0,0,0,0,0]]
alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz']]
calculate_var_explained=False
contrastCurves=[[]]
plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]

# plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_key'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,5]]
# timeShifts=[['rivalryInferredToReportedShift','replayInferredToReportedShift','rivalryInferredToReportedShift','replayInferredToReportedShift',0,0,0,0,0]]
# alignmentInfo=[['key','key','key','key','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_active_rivalry_vs_active_replay_reported_aligned_w_key'
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,5,6]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','key','tevz','tevz','tevz','key','event','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_keys_anySwitches_rivalrySwitches'
# regressors=[['keyPress_Any','inferredSwitches_Any','inferredRivalrySwitch_Any','saccades_Any','blinks_Any','trialStarts_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# contrastCurves=[[[[0,1,1,0,0,0],'RivalrySwitchAddedToAny']]]	#per GLM this lists all contrast curves. Every entry has two parts: a list of weights (position coded) for a weighted sum of response curves, and a title
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_replayedSwitches'
# regressors=[['keyPress_Any','inferredRivalrySwitch_Any','inferredReplaySwitch_Any','saccades_Any','blinks_Any','trialStarts_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_active_rivalry_inferred0duration_aligned_w_OKN'
# regressors=[['inferredSwitches0duration_Active rivalry','inferredSwitchesNon0duration_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_passive_rivalry_vs_replay_inferred_noblinks_aligned_w_OKN'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitchesNoBlink_Passive rivalry','inferredSwitchesNoBlink_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','inferredSwitchesBlink_Passive rivalry','inferredSwitchesBlink_Passive replay']]
# plottedRegressorIndices=[[2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz','OKN','OKN']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#

#and for stats, some commented out:

# plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_OKN_derivative'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,5]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_key_derivative'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,5]]
# timeShifts=[['rivalryInferredToReportedShift','replayInferredToReportedShift','rivalryInferredToReportedShift','replayInferredToReportedShift',0,0,0,0,0]]
# alignmentInfo=[['key','key','key','key','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_active_rivalry_vs_active_replay_reported_aligned_w_key_derivative'
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,5,6]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','key','tevz','tevz','tevz','key','event','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_keys_anySwitches_rivalrySwitches_derivative'
# regressors=[['keyPress_Any','inferredSwitches_Any','inferredRivalrySwitch_Any','saccades_Any','blinks_Any','trialStarts_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_replayedSwitches_derivative'
# regressors=[['keyPress_Any','inferredRivalrySwitch_Any','inferredReplaySwitch_Any','saccades_Any','blinks_Any','trialStarts_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_active_rivalry_inferred0duration_aligned_w_OKN_derivative'
# regressors=[['inferredSwitches0duration_Active rivalry','inferredSwitchesNon0duration_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#
# plotTitle='_Concatenated_passive_rivalry_vs_replay_inferred_noblinks_aligned_w_OKN_derivative'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitchesNoBlink_Passive rivalry','inferredSwitchesNoBlink_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','inferredSwitchesBlink_Passive rivalry','inferredSwitchesBlink_Passive replay']]
# plottedRegressorIndices=[[2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz','OKN','OKN']]
# calculate_var_explained=False
# contrastCurves=[[]]
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'contrastCurves':contrastCurves}]
#


#-------------
#Other stuff:
#
# plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_OKN'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,5]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_and_passive_rivalry_vs_replay_inferred_aligned_w_key'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,5]]
# timeShifts=[['rivalryInferredToReportedShift','replayInferredToReportedShift','rivalryInferredToReportedShift','replayInferredToReportedShift',0,0,0,0,0]]
# alignmentInfo=[['key','key','key','key','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_rivalry_vs_active_replay_reported_aligned_w_key'
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,5,6]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','key','tevz','tevz','tevz','key','event','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]


# temporarily off but useful otherwise
# plotTitle='_Concatenated_blinks_and_saccades'
# regressors=[['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[7,8]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','tevz','tevz','tevz','tevz','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_rivalry_reported_vs_inferred_aligned_w_OKN'
# regressors=[['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0],[0]]
# timeShifts=[['rivalryReportedToInferredShift',0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_replay_reported_vs_inferred_vs_physical_aligned_w_OKN'
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['reportedSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[1],[1],[1]]
# timeShifts=[[0,'replayReportedToInferredShift',0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,'replayPhysicalToInferredShift',0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_passive_replay_inferred_vs_physical_aligned_w_OKN'
# regressors=[['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[3],[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0],[0,0,0,'replayPhysicalToInferredShift',0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','OKN','tevz','tevz','tevz','tevz','tevz'],['tevz','tevz','tevz','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_rivalry_reported_vs_inferred_only0dur_aligned_w_OKN'
# regressors=[['reportedSwitches0duration_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','reportedSwitchStarts_Active rivalry','reportedSwitchEnds_Active rivalry'],['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0],[0]]
# timeShifts=[['rivalryReportedToInferredShift',0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_active_replay_reported_vs_inferred_vs_physical_only0dur_aligned_w_OKN'
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','reportedSwitchStarts_Active replay','reportedSwitchEnds_Active replay'],['reportedSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['reportedSwitches_Active rivalry','physicalSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchStarts_Active replay','physicalSwitchEnds_Active replay']]
# plottedRegressorIndices=[[1],[1],[1]]
# timeShifts=[[0,'replayReportedToInferredShift',0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,'replayPhysicalToInferredShift',0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz'],['tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_passive_replay_inferred_vs_physical_only0dur_aligned_w_OKN'
# regressors=[['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any'],['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchStarts_Passive replay','physicalSwitchEnds_Passive replay']]
# plottedRegressorIndices=[[3],[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0],[0,0,0,'replayPhysicalToInferredShift',0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','OKN','tevz','tevz','tevz','tevz','tevz'],['tevz','tevz','tevz','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_keys_rivarlySwitches_replayedSwitches_version1'
# regressors=[['keyPress_Any','inferredRivalrySwitch_Any','physicalSwitches0duration_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0]]
# alignmentInfo=[['key','OKN','phys','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='_Concatenated_keys_rivarlySwitches_replayedSwitches_version2'
# regressors=[['keyPress_Any','inferredRivalrySwitch_Any','physicalSwitches_Any','saccades_Any','blinks_Any']]
# plottedRegressorIndices=[[0,1,2,3,4]]
# timeShifts=[[0,0,0,0,0]]
# alignmentInfo=[['key','OKN','phys','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]

#-----------------
#and then run the GLMS

plotColors=['r','g','b','c', 'm', 'y', 'k']

for onePlotDict in plotDictList:
	
	if 'derivative' in onePlotDict['plotTitle']:
		deriveIt=True
		if 'zscore' in onePlotDict['plotTitle']:
			zscoreIt=True
			ylabel='ONZIN! Z-score of deriv. of z-score of pupil area.'	#really not necessary! z-score has already been done. Plus this z-scoring here still does something specific with within-trial samples; previous z-scoring was per trial
		else:
			zscoreIt=False	
			ylabel='Deriv. of z-score of pupil area.'
	else:
		deriveIt=False
		if 'zscore' in onePlotDict['plotTitle']:
			zscoreIt=True
			ylabel='ONZIN! z-score of z-score of pupil area.'	#really not necessary! z-score has already been done. Plus this z-scoring here still does something specific with within-trial samples; previous z-scoring was per trial
		else:
			zscoreIt=False	
			ylabel='z-score of pupil area.'
		
	individualsIncluded=[oneInfoDict['obs'] for oneInfoDict in concatenatedDictList]
	
	allDecoResponses=[]
	allStErrs=[]
	allPerObsData=[]
	
	for glmIndex in range(len(onePlotDict['regressors'])):
		
		allDecoResponsesOneGLM=[]
		
		for oneObserver in individualsIncluded:
			
			thisInfoDict=[oneInfoDict for oneInfoDict in concatenatedDictList if oneInfoDict['obs']==oneObserver][0]
			
			if deriveIt:
				paddedPupilSamplesCleaned=thisInfoDict['pupilDataDeriv']
			else:
				paddedPupilSamplesCleaned=thisInfoDict['pupilData']

			if zscoreIt:
				theseTrialStartEvents=thisInfoDict['trialStarts_Any']
				
				indicesWithinTrials=[range(int(thisTrialStartEvent*newSampleRate),int(thisTrialStartEvent*newSampleRate+downsampledSamplesPerTrial)) for thisTrialStartEvent in theseTrialStartEvents]
				indicesWithinTrials = [item for sublist in indicesWithinTrials for item in sublist]
				
				indicesWithinTrialsNew=[element for element in indicesWithinTrials if element<len(paddedPupilSamplesCleaned)]
				lengthDifference=len(indicesWithinTrials)-len(indicesWithinTrialsNew)
				if lengthDifference>0:
					print(str(lengthDifference)+' indices removed from indicesWithinTrials because they are after the data end. This is at a sample rate of '+str(newSampleRate)+' Hz.')
					indicesWithinTrials=indicesWithinTrialsNew
				
				# to verify that those are indeed the right indices
				# shell()
				# fig = pl.figure(figsize = (25,15))
				# s = fig.add_subplot(1,1,1)
				# s.set_title('Yo moma so fat')
				#
				# for plotStartIndex in range(1,len(paddedPupilSamplesCleaned)-1000,1000):
				#
				# 	includedTrialIndices=[element for element in indicesWithinTrials if element>=plotStartIndex and element<plotStartIndex+1000]
				#
				# 	pl.plot(range(plotStartIndex,plotStartIndex+1000),[paddedPupilSamplesCleaned[thisIndex] for thisIndex in range(plotStartIndex,plotStartIndex+1000)])
				# 	pl.scatter(includedTrialIndices,[0 for element in includedTrialIndices])
				#
				# 	pl.show()
				
				sigmaOfSamplesWithinTrials=numpy.std(paddedPupilSamplesCleaned[indicesWithinTrials])
				paddedPupilSamplesCleaned=paddedPupilSamplesCleaned/sigmaOfSamplesWithinTrials
				
			theseEventNames=onePlotDict['regressors'][glmIndex]
			theseEvents=[thisInfoDict[oneEventKind] for oneEventKind in theseEventNames]
			
			theseShifts=[0 if oneTimeShiftString==0 else thisInfoDict[oneTimeShiftString] for oneTimeShiftString in onePlotDict['timeShifts'][glmIndex]]
			theseEvents=theseEvents+numpy.array(theseShifts)		#align the timing of reported and physical switches with that of the inferred ones
			
			eventTypesIndicesIncluded=[]
			for oneEventTypeIndex in range(len(theseEventNames)):
				if len(theseEvents[oneEventTypeIndex])>minNumEvs:
					eventTypesIndicesIncluded=eventTypesIndicesIncluded+[oneEventTypeIndex]
				else:
					print 'Event named '+theseEventNames[oneEventTypeIndex]+' excluded from '+onePlotDict['plotTitle']+' for observer '+oneObserver+' because insufficient events'
			
			theseEventsActuallyUsed=[theseEvents[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]			
			theseEventNamesActuallyUsed=[theseEventNames[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]			
			
			#------------
			rfy = nideconv.ResponseFitter(input_signal=paddedPupilSamplesCleaned,sample_rate=newSampleRate)

			for eventIndex,eventName in enumerate(theseEventNamesActuallyUsed):
				if eventName == 'saccades_Any':
					thisInterval=decoIntervalSacc
					numFuncs=numBasisFunctionsIncOffsetSacc
				elif eventName == 'blinks_Any':
					thisInterval=decoIntervalBlink
					numFuncs=numBasisFunctionsIncOffsetBlink
				else:
					thisInterval=decoInterval
					numFuncs=numBasisFunctionsIncOffset
										
				rfy.add_event(event_name=eventName,onset_times=theseEventsActuallyUsed[eventIndex],basis_set='fourier',interval=thisInterval,n_regressors=numFuncs)
				
			rfy.regress()
			myIRFs=rfy.get_timecourses()
			#------------
			
			decoResponsesOneGLMAndObsInCorrectOrder=[]
			
			for oneEventName in theseEventNames:
				if not oneEventName in theseEventNamesActuallyUsed:
					decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]]
				else:	
					decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[list(myIRFs[0][oneEventName])]
			
			allContrastCurves=[]
			for contrastCurveInfo in onePlotDict['contrastCurves'][glmIndex]:
				
				weights=contrastCurveInfo[0]
				
				summedCurve=numpy.zeros(len(decoResponsesOneGLMAndObsInCorrectOrder[0]))
				for oneResponseIndex,oneResponse in enumerate(decoResponsesOneGLMAndObsInCorrectOrder):
					if not(weights[oneResponseIndex]==0):
						summedCurve=summedCurve+numpy.array(oneResponse)*weights[oneResponseIndex]
				
				allContrastCurves=allContrastCurves+[list(summedCurve)]
				
			decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+allContrastCurves
			allDecoResponsesOneGLM=allDecoResponsesOneGLM+[decoResponsesOneGLMAndObsInCorrectOrder]

		theAverage=[]
		theStErr=[]
		thePerObsData=[]
		for regressorIndex in range(len(allDecoResponsesOneGLM[0])):
			onlyIncludedObservers=[allDecoResponsesOneGLM[obsIndex][regressorIndex] for obsIndex in range(len(allDecoResponsesOneGLM)) if not (min(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1 and max(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1)]
			if onlyIncludedObservers==[]:
				thisAverage=[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]
			else:
				thisAverage=numpy.average(onlyIncludedObservers,0)
				thisStErr=numpy.std(onlyIncludedObservers,0)/float(numpy.sqrt(len(onlyIncludedObservers)))
			theAverage=theAverage+[thisAverage]
			theStErr=theStErr+[thisStErr]		#theStErr is all the across-obs st Errs (one per regressor) within this GLM
			thePerObsData=thePerObsData+[onlyIncludedObservers]	#thePerObsData is, for this GLM, all the individual-observer data for included observers only. Nesting is: all regressors, all included observers, all timepoints

		allDecoResponsesOneGLM=allDecoResponsesOneGLM+[theAverage]	#add theAverage as if it's just another participant
		allDecoResponses=allDecoResponses+[allDecoResponsesOneGLM]
		allStErrs=allStErrs+[theStErr]		#allStErrs is bunch of theStErr's, one for each GLM
		allPerObsData=allPerObsData+[thePerObsData]		#allPerObsData is bunch of thePerObsData's, one for each GLM. So nesting is: GLM, regressor, observer (only included ones), timepoint. It's very similar to allDecoResponsesOneGLM but nested in a different order and with individual observers removed if their data didn't include a particular regressor.

		if len(onePlotDict['contrastCurves'][glmIndex])>0:		#add names and alignment info for contrast curves for plotting
			onePlotDict['plottedRegressorIndices'][glmIndex]=onePlotDict['plottedRegressorIndices'][glmIndex]+[len(onePlotDict['regressors'][glmIndex])+thisIndex for thisIndex in range(len(onePlotDict['contrastCurves'][glmIndex]))]
			onePlotDict['regressors'][glmIndex]=onePlotDict['regressors'][glmIndex]+[element[1] for element in onePlotDict['contrastCurves'][glmIndex]]
			onePlotDict['alignmentInfo'][glmIndex]=onePlotDict['alignmentInfo'][glmIndex]+['n/a' for element in onePlotDict['contrastCurves'][glmIndex]]
			
#	x = numpy.linspace(decoInterval[0],decoInterval[1], len(allDecoResponses[0][0][0]))

	forOutputAccompanyingPlot=[]
	
	individualsIncludedPlusAverage=individualsIncluded+['Average']
	f = pl.figure(figsize = (35,35))
	for observerIndex in range(len(individualsIncluded)+1):
		
		forOutputAccompanyingPlotOneObs=[]
		
		s=f.add_subplot(5,6,observerIndex+1)
		colorCounter=0
		for glmIndex in range(len(onePlotDict['regressors'])):
			theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]

			for regressorIndex in theseRegressorIndices:
				
				y=allDecoResponses[glmIndex][observerIndex][regressorIndex]
				forOutputAccompanyingPlotOneObs=forOutputAccompanyingPlotOneObs+[y]
				
				regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]
				
				if 'blinks' in regressorName:
					x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
				elif 'saccades' in regressorName:
					x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
				else:
					x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
				
				pl.plot(x, y, color=plotColors[colorCounter], label=regressorName)
				colorCounter=colorCounter+1

		if not observerIndex==len(individualsIncluded):
			forOutputAccompanyingPlot=forOutputAccompanyingPlot+[forOutputAccompanyingPlotOneObs]
		
		pl.xlabel('Time from event (s)')
		pl.ylabel(ylabel)
		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
		#sn.despine(offset=10)
		s.set_title(individualsIncludedPlusAverage[observerIndex])

	pl.legend(loc=2)

	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
	colorCounter=0
	for glmIndex in range(len(onePlotDict['regressors'])):
		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
		for regressorIndex in theseRegressorIndices:
			y=allDecoResponses[glmIndex][-1][regressorIndex]		#-1 will be the across-obs average
			stErrs=allStErrs[glmIndex][regressorIndex]

			regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]
		
			if 'blinks' in regressorName:
				x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
			elif 'saccades' in regressorName:
				x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
			else:
				x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
		
			pl.plot(x, y, color=plotColors[colorCounter], label=regressorName)
			
			downSampledY=[y[index] for index in range(0,len(y),40)]
			downSampledX=[x[index] for index in range(0,len(y),40)]
			downSampledErr=[stErrs[index] for index in range(0,len(y),40)]
			
			pl.errorbar(downSampledX, downSampledY, yerr=downSampledErr, color=plotColors[colorCounter], ls='none')
			
			tTestpValsVs0=[ttest_1samp([allPerObsData[glmIndex][regressorIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(0,len(y),40)]
			
			xForSignificantOnes=[]
			yForSignificantOnes=[]
			for candidateIndex in range(len(downSampledY)):
				if tTestpValsVs0[candidateIndex]<.01:
					xForSignificantOnes=xForSignificantOnes+[downSampledX[candidateIndex]]
					yForSignificantOnes=yForSignificantOnes+[downSampledY[candidateIndex]]

			pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)

			colorCounter=colorCounter+1

	pl.xlabel('Time from event (s)')
	pl.ylabel(ylabel)
	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
	#sn.despine(offset=10)
	s.set_title('Average plus error bars')

	s=f.add_subplot(5,6,observerIndex+3)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
	colorCounter=0
	for glmIndex in range(len(onePlotDict['regressors'])):
		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
		for regressorIndex in theseRegressorIndices:
			y=allDecoResponses[glmIndex][-1][regressorIndex]		#-1 will be the across-obs average
			stErrs=allStErrs[glmIndex][regressorIndex]

			regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]
		
			if 'blinks' in regressorName:
				x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
			elif 'saccades' in regressorName:
				x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
			else:
				x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
		
			pl.plot(x, y, color=plotColors[colorCounter], label=regressorName)
			pl.plot(x, [y[thisIndex]-stErrs[thisIndex] for thisIndex in range(len(y))], color=plotColors[colorCounter], linewidth=1)
			pl.plot(x, y, color=plotColors[colorCounter], label=regressorName)
			pl.plot(x, [y[thisIndex]+stErrs[thisIndex] for thisIndex in range(len(y))], color=plotColors[colorCounter], linewidth=1)
			
			colorCounter=colorCounter+1

	pl.xlabel('Time from event (s)')
	pl.ylabel(ylabel)
	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
	#sn.despine(offset=10)
	s.set_title('Average plus error bars')

	pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_nideconv.pdf')
	numpy.save(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_nideconv_thedata',forOutputAccompanyingPlot)
	
		
