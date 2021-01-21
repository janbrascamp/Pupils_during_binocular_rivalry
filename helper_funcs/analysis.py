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
import scipy.signal
import scipy.optimize
import pickle
from sklearn import linear_model

import matplotlib
import matplotlib.pyplot as pl

from . import commandline
from . import filemanipulation
from . import userinteraction
from . import utilities

from IPython import embed as shell

def doNotFilterButDoCleanPupilSize(preprocessInfoDict,inputPath,outputPathFiltered,outputPathRegressors,inputFileName,outputFileName):
	
	commandline.ShellCmd('mkdir '+outputPathFiltered)
	commandline.ShellCmd('mkdir '+outputPathRegressors)
	filteredFilesPresent=commandline.putFileNamesInArray(outputPathFiltered)
	
	if not(outputFileName+'_cleaned_pupil.txt' in filteredFilesPresent):
		
		userinteraction.printFeedbackMessage("cleaning up pupil data for "+outputFileName)
	
		sampleData=filemanipulation.readDelimitedIntoArray(inputPath+inputFileName+'_s.asc','\t')
		
		sampleData=[[float(rowElement) if rowElement.strip(' .CR') else -1.0 for rowElement in row] for row in sampleData]	#if only contains spaces, dots C's and/or R's, then will evaluate to false because empty string after stripping. This is applied to individual elements in the rows at a time; not whole rows
	
		filename=inputPath+inputFileName+'_e.asc'
		f = open(filename, 'r')
		rawtext=f.read()
		f.close()
		
		#shift all time points relative to event defined in 'timeZeroInfo', and delete whatever comes before
		regExp=preprocessInfoDict['timeZeroInfo'][0]
		rankNumber=preprocessInfoDict['timeZeroInfo'][1]
		timeZero=int(re.findall(regExp,rawtext)[rankNumber])
		timeZeroIncBuffer=timeZero-(preprocessInfoDict['timeZeroInfo'][2])*1000.
		
		sampleData=[[element[0]-timeZero,numpy.average([element[1],element[4]]),numpy.average([element[2],element[5]]),numpy.average([element[3],element[6]])] for element in sampleData if element[0]>=timeZeroIncBuffer]		#this is binocular: 6 element besides moment; would otherwise be 3. immediately average eyes. x and y gaze also contained in the resulting list but not actually used.
		
		timePerRowAscFile_ms=preprocessInfoDict['timePerRowAscFile_s']*1000.
		
		allSampleIntervals=[sampleData[index+1][0]-sampleData[index][0] for index in range(len(sampleData)-1)] #in this experiment recording can be switched off during the experiment to re-calibrate. Later analysis assumes that the samples are uninterrupted. So pad any missing samples and treat them as a blink.
		allUniqueSampleIntervals=list(set().union(allSampleIntervals))#because the procedure below is so slow, quickly test whether it's necessary first
		
		intervalsTreatedAsBlink_ms=[]
		if len(allUniqueSampleIntervals)>1:
			userinteraction.printFeedbackMessage("Beware. Interpolating missing data due to recalibration. This can involve shifting data points in time by up to "+str(timePerRowAscFile_ms)+" ms.")
			
			sampleData=[[element[0]-numpy.mod(element[0],timePerRowAscFile_ms)]+element[1:] for element in sampleData]	#if there has been an interruption, then the tracker might not have resumed a multiple of timePerRowAscFile_ms after the start of the initial recording. We do want all consecutive samples to be timePerRowAscFile_ms apart, so we shift the data in time ever so slightly]
			
			allBlockRightBorders=[index for index,value in enumerate(allSampleIntervals) if not value==timePerRowAscFile_ms]+[len(sampleData)-1]
			
			allBlocks=[]
			oneLeftBorder=0
			for blockIndex, oneRightBorder in enumerate(allBlockRightBorders):
				
				allBlocks=allBlocks+[sampleData[index] for index in range(oneLeftBorder,oneRightBorder+1)]
				
				if not blockIndex==(len(allBlockRightBorders)-1):
					
					insertTimepoints=numpy.linspace(sampleData[oneRightBorder][0]+timePerRowAscFile_ms,sampleData[oneRightBorder+1][0]-timePerRowAscFile_ms,int((sampleData[oneRightBorder+1][0]-sampleData[oneRightBorder][0]-2.*timePerRowAscFile_ms)/timePerRowAscFile_ms))
					allBlocks=allBlocks+[[oneTimepoint,-1.,-1.,-1.] for oneTimepoint in insertTimepoints]	#insert missing data, thereby turning it into a continuous time axis
					intervalsTreatedAsBlink_ms=intervalsTreatedAsBlink_ms+[[sampleData[oneRightBorder][0]+timePerRowAscFile_ms,sampleData[oneRightBorder+1][0]-timePerRowAscFile_ms]]	#this does not, actually, seem to set a blink regressor halfway that gap; it's marking the start and end, and later the start point is saved to regressor file
				
				oneLeftBorder=oneRightBorder+1
				
			sampleData=allBlocks[:]	#not sure why making copy instead of adding new name to same list. tevz.
			
		else:
			
			userinteraction.printFeedbackMessage("No recalibration occurred. We continue.")
			
		#linear interpolation of blinks, either eye. If blinks for the two eyes overlap (often, then combine earliest starting point and latest endpoint to create a conservative hyperblink)
		regExp='EBLINK'
		
		blinkIntervalsInitial_ms=[[float(rowElement)-timeZero for rowElement in row] for row in re.findall(regExp+' [LR] ([0-9]*)\t *([0-9]*)',rawtext)]
		
		blinkIntervalsInitial_ms=blinkIntervalsInitial_ms+intervalsTreatedAsBlink_ms
		blinkIntervalsInitial_ms.sort()

		blinkNum=len(blinkIntervalsInitial_ms)
		blinkIntervals_ms=[]
		thisIndex=0
		while  True:
			
			thisBlink=blinkIntervalsInitial_ms[thisIndex]
			nextIndexWalker=1
			try:
				nextBlink=blinkIntervalsInitial_ms[thisIndex+nextIndexWalker]
				while nextBlink[0]<thisBlink[1]:	#if next one started before this one ends...
					thisBlink[0]=min([thisBlink[0],nextBlink[0]])
					thisBlink[1]=max([thisBlink[1],nextBlink[1]])		#...then use earliest start and latest end, so you get a larger hyperblink
					nextIndexWalker=nextIndexWalker+1
					nextBlink=blinkIntervalsInitial_ms[thisIndex+nextIndexWalker]	#can even glue multiple blinks together
			except IndexError:
				if (thisIndex+nextIndexWalker)==blinkNum:
					blinkIntervals_ms=blinkIntervals_ms+[thisBlink]
					break
				else:
					raise IndexError
			
			blinkIntervals_ms=blinkIntervals_ms+[thisBlink]
			thisIndex=thisIndex+nextIndexWalker
		
		blinkStart_secondsWithinScan=[blinkInterval_ms[0]/1000. for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]	#think 'session' when it says scan
		blinkStartAndDuration_secondsWithinScan=[[blinkInterval_ms[0]/1000.,(blinkInterval_ms[1]-blinkInterval_ms[0])/1000.] for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]
		blink_standardRegressorFormat=[[numpy.average([blinkInterval_ms[0],blinkInterval_ms[1]])/1000.,1.] for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]
		
		numpy.savetxt(outputPathRegressors+outputFileName+'_blink_moments.txt',numpy.array(blinkStart_secondsWithinScan),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors
		numpy.savetxt(outputPathRegressors+outputFileName+'_blink_moments_and_durations.txt',numpy.array(blinkStartAndDuration_secondsWithinScan),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors
		numpy.savetxt(outputPathRegressors+outputFileName+'_blinkMomentsStandardRegressorFormat.txt',numpy.array(blink_standardRegressorFormat),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors. This one has the same format (i.e. moment, value) as the behavioral regressors will have

		blinkFlankingInterval_samples=int(preprocessInfoDict['blinkFlankDurInterpolation_s']/preprocessInfoDict['timePerRowAscFile_s'])
		blinkSideBuffer_samples=int(preprocessInfoDict['blinkSideBuffer_s']/preprocessInfoDict['timePerRowAscFile_s'])
	
		firstTimePointSfile_ms=sampleData[0][0]		#so this is basically negative the buffer
		minChunkDur_ms=preprocessInfoDict['minChunkDur_s']*1000.
		maxBlinkDur_ms=preprocessInfoDict['maxBlinkDur_s']*1000.
		
		lengthSampleArray=len(sampleData)
		pupilColumn=3
		
		sampleDataBeforeInterpolation=[element for element in sampleData]
		
		deleteIntervals=[]
		for blinkInterval_ms in blinkIntervals_ms:	#I believe this is the code that expects all sample times to be present (even if with missing data), because this code uses indices into the array instead of time
			
			interpolate=True
			
			frontFlankIndex_A=int((blinkInterval_ms[0]-firstTimePointSfile_ms)/timePerRowAscFile_ms)-blinkSideBuffer_samples
			frontFlankIndices=[max(frontFlankIndex_A-blinkFlankingInterval_samples,0),max(frontFlankIndex_A+1,0)]
	
			behindFlankIndex_A=int((blinkInterval_ms[1]-firstTimePointSfile_ms)/timePerRowAscFile_ms)+blinkSideBuffer_samples
			behindFlankIndices=[min(behindFlankIndex_A,lengthSampleArray),min(behindFlankIndex_A+blinkFlankingInterval_samples+1,lengthSampleArray)]
			
			if frontFlankIndices[1]==0:		#there is no interval in front of this 'blink' because it's at the beginning of the entire run
				interpolate=False
				deleteIntervals=deleteIntervals+range(0,behindFlankIndices[0])
			elif behindFlankIndices[0]==lengthSampleArray:		#there is no interval at the end of this 'blink' because it's at the end of the entire run
				interpolate=False
				deleteIntervals=deleteIntervals+range(frontFlankIndices[1],lengthSampleArray)
			
			# interpolate regardless of whether this is an actual blink or probable signal drop
			# elif (behindFlankIndices[0]-frontFlankIndices[1])>(maxBlinkDur_ms/timePerRowAscFile_ms):	#if the 'blink' is longer than maxBlinkDur_s then it's missing data rather than an actual blink and it doesn't make much sense to interpolate it, and deleting the data is better
			# 	interpolate=False
			# 	deleteIntervals=deleteIntervals+range(frontFlankIndices[1],behindFlankIndices[0])
				
			if interpolate:
				meanFront=numpy.mean([sampleData[index][pupilColumn] for index in range(frontFlankIndices[0],frontFlankIndices[1])])
				meanBehind=numpy.mean([sampleData[index][pupilColumn] for index in range(behindFlankIndices[0],behindFlankIndices[1])])
				meanDifference=meanBehind-meanFront
				blinkDurPlusBuffers=behindFlankIndices[0]-frontFlankIndices[1]
		
				for i,replacementIndex in enumerate(range(frontFlankIndices[1],behindFlankIndices[0])):
					sampleData[replacementIndex][pupilColumn]=meanFront+meanDifference*(i+1)/blinkDurPlusBuffers
	
		deleteIntervalsSet=set(deleteIntervals)	#faster lookup in next row
		frameNumbersAndPupilSizes=[[sampleData[index][0],sampleData[index][pupilColumn]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		
		frameNumbersAndPupilSizesBeforeInterpolation=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][pupilColumn]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		frameNumbersAndGazeXpoints=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][1]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		frameNumbersAndGazeYpoints=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][2]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		
		#NO FILTERING HERE: DO IT AFTER ALL DEAD TIME (INTER-TRIAL PERIODS) has been removed
		#new approach:
		#there should be no gaps in the time series now (everything has been interpolated) so just filter entire thing at once. also don't remove outliers because it causes gaps again
		#nyquistFreq=(1./preprocessInfoDict['timePerRowAscFile_s'])/2.
		#criticalFreqs=[freq/nyquistFreq for freq in preprocessInfoDict['filterCutoffs_Hz']]	#from http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.butter.html: For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
		#
		#b,a=scipy.signal.butter(3, criticalFreqs[1],btype='low')		#filtering low-pass and high-pass consecutively because bandpass doesn't work for some reason
		#filteredPupilSizesTemp=scipy.signal.filtfilt(b,a,[element[1] for element in frameNumbersAndPupilSizes])
		#b,a=scipy.signal.butter(3, criticalFreqs[0],btype='high')
		#filteredPupilSizes=scipy.signal.filtfilt(b,a,filteredPupilSizesTemp)
		#
		#frameNumbersAndPupilSizes=[[frameNumbersAndPupilSizes[index][0],filteredPupilSizes[index]] for index in range(len(filteredPupilSizes))]
		
		#z-score per session. No, let's just 0-mean: the eye tracker is always at similar distance from the eye.
		#This is not important, because the whole thing is 0-meaned again after setting the data that falls between trials to 0,
		#but it's not a problem either.
		
		meanPupilSize=numpy.mean([row[1] for row in frameNumbersAndPupilSizes])
		#sigmaPupilSize=numpy.std([row[1] for row in frameNumbersAndPupilSizes])
		
		frameNumbersAndPupilSizes=[[frameNumbersAndPupilSizes[index][0],(frameNumbersAndPupilSizes[index][1]-meanPupilSize)] for index in range(len(frameNumbersAndPupilSizes))] #/sigmaPupilSize] for index in range(len(frameNumbersAndPupilSizes))]
		
		numpy.savetxt(outputPathFiltered+outputFileName+'_cleaned_pupil.txt',numpy.array(frameNumbersAndPupilSizes),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_non_interpolated_pupil.txt',numpy.array(frameNumbersAndPupilSizesBeforeInterpolation),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_xGaze_on_pupil_axis.txt',numpy.array(frameNumbersAndGazeXpoints),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_yGaze_on_pupil_axis.txt',numpy.array(frameNumbersAndGazeYpoints),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers
			
	else:
		
		userinteraction.printFeedbackMessage("not cleaning up pupil data for "+outputFileName+" because already done")
		
def doNotFilterButDoCleanPupilSizeAfterTomasAndGillesInputSept2020(preprocessInfoDict,inputPath,outputPathFiltered,outputPathRegressors,inputFileName,outputFileName):
	
	commandline.ShellCmd('mkdir '+outputPathFiltered)
	commandline.ShellCmd('mkdir '+outputPathRegressors)
	filteredFilesPresent=commandline.putFileNamesInArray(outputPathFiltered)
	
	if not(outputFileName+'_cleaned_pup_GT0920.txt' in filteredFilesPresent):
		
		userinteraction.printFeedbackMessage("cleaning up pupil data for "+outputFileName)
	
		sampleData=filemanipulation.readDelimitedIntoArray(inputPath+inputFileName+'_s.asc','\t')
		
		sampleData=[[float(rowElement) if rowElement.strip(' .CR') else -1.0 for rowElement in row] for row in sampleData]	#if only contains spaces, dots C's and/or R's, then will evaluate to false because empty string after stripping. This is applied to individual elements in the rows at a time; not whole rows
	
		filename=inputPath+inputFileName+'_e.asc'
		f = open(filename, 'r')
		rawtext=f.read()
		f.close()
		
		#shift all time points relative to event defined in 'timeZeroInfo', and delete whatever comes before
		regExp=preprocessInfoDict['timeZeroInfo'][0]
		rankNumber=preprocessInfoDict['timeZeroInfo'][1]
		timeZero=int(re.findall(regExp,rawtext)[rankNumber])
		timeZeroIncBuffer=timeZero-(preprocessInfoDict['timeZeroInfo'][2])*1000.
		
		sampleData=[[element[0]-timeZero,numpy.average([element[1],element[4]]),numpy.average([element[2],element[5]]),numpy.average([element[3],element[6]])] for element in sampleData if element[0]>=timeZeroIncBuffer]		#this is binocular: 6 element besides moment; would otherwise be 3. immediately average eyes. x and y gaze also contained in the resulting list but not actually used.
		
		timePerRowAscFile_ms=preprocessInfoDict['timePerRowAscFile_s']*1000.
		
		allSampleIntervals=[sampleData[index+1][0]-sampleData[index][0] for index in range(len(sampleData)-1)] #in this experiment recording can be switched off during the experiment to re-calibrate. Later analysis assumes that the samples are uninterrupted. So pad any missing samples and treat them as a blink.
		allUniqueSampleIntervals=list(set().union(allSampleIntervals))#because the procedure below is so slow, quickly test whether it's necessary first
		
		intervalsInsertedDuringRecalibration_ms=[]
		if len(allUniqueSampleIntervals)>1:
			userinteraction.printFeedbackMessage("Beware. Interpolating missing data due to recalibration. This can involve shifting data points in time by up to "+str(timePerRowAscFile_ms)+" ms.")
			
			sampleData=[[element[0]-numpy.mod(element[0],timePerRowAscFile_ms)]+element[1:] for element in sampleData]	#if there has been an interruption, then the tracker might not have resumed a multiple of timePerRowAscFile_ms after the start of the initial recording. We do want all consecutive samples to be timePerRowAscFile_ms apart, so we shift the data in time ever so slightly]
			
			allBlockRightBorders=[index for index,value in enumerate(allSampleIntervals) if not value==timePerRowAscFile_ms]+[len(sampleData)-1]
			
			allBlocks=[]
			oneLeftBorder=0
			for blockIndex, oneRightBorder in enumerate(allBlockRightBorders):
				
				allBlocks=allBlocks+[sampleData[index] for index in range(oneLeftBorder,oneRightBorder+1)]
				
				if not blockIndex==(len(allBlockRightBorders)-1):
					
					insertTimepoints=numpy.linspace(sampleData[oneRightBorder][0]+timePerRowAscFile_ms,sampleData[oneRightBorder+1][0]-timePerRowAscFile_ms,int((sampleData[oneRightBorder+1][0]-sampleData[oneRightBorder][0]-2.*timePerRowAscFile_ms)/timePerRowAscFile_ms))
					allBlocks=allBlocks+[[oneTimepoint,-66.,-66.,-66.] for oneTimepoint in insertTimepoints]	#insert missing data, thereby turning it into a continuous time axis
					
					#ALTERED IN RESPONSE TO INTERACTION WITH TOMAS AND GILLES, SEPT 2020: RECALIBRATION PERIODS NO LONGER INTERPOLATED AS IF THEY WERE BLINKS. INSTEAD THE -66 STAYS THERE TILL AFTER FILTERING AND Z-SCORING (PER RECORDING BLOCK; EXCLUDING THE PADDED PART ADDED ABOVE WITH -66s)
					intervalsInsertedDuringRecalibration_ms=intervalsInsertedDuringRecalibration_ms+[[sampleData[oneRightBorder][0]+timePerRowAscFile_ms,sampleData[oneRightBorder+1][0]-timePerRowAscFile_ms]]	#this marks the start and end of what has been inserted, so that it can be stored and later filtering and z-scoring etc. can take place on the basis of actual data rather than this interpolated stuff
				
				oneLeftBorder=oneRightBorder+1
				
			sampleData=allBlocks[:]	#not sure why making copy instead of adding new name to same list. tevz.
			
			
			recalibrationsStartAndEnd_secondsWithinScan=[[oneInterval[0]/1000.,oneInterval[1]/1000.] for oneInterval in intervalsInsertedDuringRecalibration_ms]
			numpy.savetxt(outputPathRegressors+outputFileName+'_recalibration_periods.txt',numpy.array(recalibrationsStartAndEnd_secondsWithinScan),fmt='%20.10f')	#units are seconds. to be used later to avoid those time periods during filtering and z-scoring
		
		else:
			
			numpy.savetxt(outputPathRegressors+outputFileName+'_recalibration_periods.txt',numpy.array([]),fmt='%20.10f')	#units are seconds. to be used later to avoid those time periods during filtering and z-scoring
			userinteraction.printFeedbackMessage("No recalibration occurred. We continue.")
			
		#linear interpolation of blinks, either eye. If blinks for the two eyes overlap (often, then combine earliest starting point and latest endpoint to create a conservative hyperblink)
		regExp='EBLINK'
		
		blinkIntervalsInitial_ms=[[float(rowElement)-timeZero for rowElement in row] for row in re.findall(regExp+' [LR] ([0-9]*)\t *([0-9]*)',rawtext)]
		
		#ALTERED IN RESPONSE TO INTERACTION WITH TOMAS AND GILLES, SEPT 2020:
		#blinkIntervalsInitial_ms=blinkIntervalsInitial_ms+intervalsTreatedAsBlink_ms
		#blinkIntervalsInitial_ms.sort()

		blinkNum=len(blinkIntervalsInitial_ms)
		blinkIntervals_ms=[]
		thisIndex=0
		while  True:
			
			thisBlink=blinkIntervalsInitial_ms[thisIndex]
			nextIndexWalker=1
			try:
				nextBlink=blinkIntervalsInitial_ms[thisIndex+nextIndexWalker]
				while nextBlink[0]<thisBlink[1]:	#if next one started before this one ends...
					thisBlink[0]=min([thisBlink[0],nextBlink[0]])
					thisBlink[1]=max([thisBlink[1],nextBlink[1]])		#...then use earliest start and latest end, so you get a larger hyperblink
					nextIndexWalker=nextIndexWalker+1
					nextBlink=blinkIntervalsInitial_ms[thisIndex+nextIndexWalker]	#can even glue multiple blinks together
			except IndexError:
				if (thisIndex+nextIndexWalker)==blinkNum:
					blinkIntervals_ms=blinkIntervals_ms+[thisBlink]
					break
				else:
					raise IndexError
			
			blinkIntervals_ms=blinkIntervals_ms+[thisBlink]
			thisIndex=thisIndex+nextIndexWalker
		
		blinkStart_secondsWithinScan=[blinkInterval_ms[0]/1000. for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]	#think 'session' when it says scan
		blinkStartAndDuration_secondsWithinScan=[[blinkInterval_ms[0]/1000.,(blinkInterval_ms[1]-blinkInterval_ms[0])/1000.] for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]
		blink_standardRegressorFormat=[[numpy.average([blinkInterval_ms[0],blinkInterval_ms[1]])/1000.,1.] for blinkInterval_ms in blinkIntervals_ms if (blinkInterval_ms[0]/1000.)>=-(preprocessInfoDict['timeZeroInfo'][2])]
		
		numpy.savetxt(outputPathRegressors+outputFileName+'_blink_moments.txt',numpy.array(blinkStart_secondsWithinScan),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors
		numpy.savetxt(outputPathRegressors+outputFileName+'_blink_moments_and_durations.txt',numpy.array(blinkStartAndDuration_secondsWithinScan),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors
		numpy.savetxt(outputPathRegressors+outputFileName+'_blinkMomentsStandardRegressorFormat.txt',numpy.array(blink_standardRegressorFormat),fmt='%20.10f')	#units are seconds. to be used later to create blink regressors. This one has the same format (i.e. moment, value) as the behavioral regressors will have
		
		blinkFlankingInterval_samples=int(preprocessInfoDict['blinkFlankDurInterpolation_s']/preprocessInfoDict['timePerRowAscFile_s'])
		blinkSideBufferPre_samples=int(preprocessInfoDict['blinkSideBufferPre_s']/preprocessInfoDict['timePerRowAscFile_s'])
		blinkSideBufferPost_samples=int(preprocessInfoDict['blinkSideBufferPost_s']/preprocessInfoDict['timePerRowAscFile_s'])
		
		firstTimePointSfile_ms=sampleData[0][0]		#so this is basically negative the buffer (not the blink buffer but the (2 s) buffer at the start
		minChunkDur_ms=preprocessInfoDict['minChunkDur_s']*1000.
		#maxBlinkDur_ms=preprocessInfoDict['maxBlinkDur_s']*1000.
		
		lengthSampleArray=len(sampleData)
		pupilColumn=3
		
		sampleDataBeforeInterpolation=[element for element in sampleData]	#apparently even this doesn't make a new copy but just a pointer so the 'before interpolation stuff' isn't actually before interpolation. need to make a deepcopy if I care about it.
		
		deleteIntervals=[]
		for blinkIndex,blinkInterval_ms in enumerate(blinkIntervals_ms):	#I believe this is the code that expects all sample times to be present (even if with missing data), because this code uses indices into the array instead of time
			
			interpolate=True
			
			frontFlankIndex_A=int((blinkInterval_ms[0]-firstTimePointSfile_ms)/timePerRowAscFile_ms)-blinkSideBufferPre_samples
			frontFlankIndices=[max(frontFlankIndex_A-blinkFlankingInterval_samples,0),max(frontFlankIndex_A+1,0)]
	
			behindFlankIndex_A=int((blinkInterval_ms[1]-firstTimePointSfile_ms)/timePerRowAscFile_ms)+blinkSideBufferPost_samples
			behindFlankIndices=[min(behindFlankIndex_A,lengthSampleArray),min(behindFlankIndex_A+blinkFlankingInterval_samples+1,lengthSampleArray)]
			
			if frontFlankIndices[1]==0:		#there is no interval in front of this 'blink' because it's at the beginning of the entire run
				interpolate=False
				deleteIntervals=deleteIntervals+range(0,behindFlankIndices[0])
			elif behindFlankIndices[0]==lengthSampleArray:		#there is no interval at the end of this 'blink' because it's at the end of the entire run
				interpolate=False
				deleteIntervals=deleteIntervals+range(frontFlankIndices[1],lengthSampleArray)
			
			# interpolate regardless of whether this is an actual blink or probable signal drop
			# elif (behindFlankIndices[0]-frontFlankIndices[1])>(maxBlinkDur_ms/timePerRowAscFile_ms):	#if the 'blink' is longer than maxBlinkDur_s then it's missing data rather than an actual blink and it doesn't make much sense to interpolate it, and deleting the data is better
			# 	interpolate=False
			# 	deleteIntervals=deleteIntervals+range(frontFlankIndices[1],behindFlankIndices[0])
				
			if interpolate:
				
				if blinkIndex<(len(blinkIntervals_ms)-1):		#Added Oct 2020: make sure you don't interpolate using data that falls inside the next blink
					nextBlinkInterval_ms=blinkIntervals_ms[blinkIndex+1]
					nextBlinkFrontFlankIndex_A=int((nextBlinkInterval_ms[0]-firstTimePointSfile_ms)/timePerRowAscFile_ms)-blinkSideBufferPre_samples
					nextBlinkBehindFlankIndex_A=int((nextBlinkInterval_ms[1]-firstTimePointSfile_ms)/timePerRowAscFile_ms)+blinkSideBufferPost_samples
					nextBlinkInterpolatedIndices=range(nextBlinkFrontFlankIndex_A,nextBlinkBehindFlankIndex_A)
					
				else:
					nextBlinkInterpolatedIndices=[]
					
				indicesForMeanFront=[index for index in range(frontFlankIndices[0],frontFlankIndices[1])]
				meanFront=numpy.mean([sampleData[index][pupilColumn] for index in indicesForMeanFront])
				
				indicesForMeanBehind=[index for index in range(behindFlankIndices[0],behindFlankIndices[1]) if not index in nextBlinkInterpolatedIndices]
				
				if indicesForMeanBehind==[]:
					meanBehind=meanFront			#if the whole interval from which the after-blink average is taken, falls inside the next blink, then just continue the before-blink average across this entire blink
					print('No post-blink interval available for interpolation because that\'s where the next blink is! This is where we are: '+outputFileName+'; blink index '+str(blinkIndex))
				else:
					meanBehind=numpy.mean([sampleData[index][pupilColumn] for index in indicesForMeanBehind])
				
				if indicesForMeanFront==[]:
					print('This should not be possible. Problem!')
					shell()
				
				meanDifference=meanBehind-meanFront
				blinkDurPlusBuffers=behindFlankIndices[0]-frontFlankIndices[1]
		
				for i,replacementIndex in enumerate(range(frontFlankIndices[1],behindFlankIndices[0])):
					sampleData[replacementIndex][pupilColumn]=meanFront+meanDifference*(i+1)/blinkDurPlusBuffers
	
		deleteIntervalsSet=set(deleteIntervals)	#faster lookup in next row
		frameNumbersAndPupilSizes=[[sampleData[index][0],sampleData[index][pupilColumn]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]	#REALLY MISNAMED 'FRAMENUMBERS': THIS IS MILLISECONDS
		
		frameNumbersAndPupilSizesBeforeInterpolation=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][pupilColumn]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		frameNumbersAndGazeXpoints=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][1]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		frameNumbersAndGazeYpoints=[[sampleDataBeforeInterpolation[index][0],sampleDataBeforeInterpolation[index][2]] for index in range(lengthSampleArray) if not index in deleteIntervalsSet]
		
		#NO FILTERING HERE: DO IT AFTER ALL DEAD TIME (INTER-TRIAL PERIODS) has been removed
		#new approach:
		#there should be no gaps in the time series now (everything has been interpolated) so just filter entire thing at once. also don't remove outliers because it causes gaps again
		#nyquistFreq=(1./preprocessInfoDict['timePerRowAscFile_s'])/2.
		#criticalFreqs=[freq/nyquistFreq for freq in preprocessInfoDict['filterCutoffs_Hz']]	#from http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.butter.html: For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
		#
		#b,a=scipy.signal.butter(3, criticalFreqs[1],btype='low')		#filtering low-pass and high-pass consecutively because bandpass doesn't work for some reason
		#filteredPupilSizesTemp=scipy.signal.filtfilt(b,a,[element[1] for element in frameNumbersAndPupilSizes])
		#b,a=scipy.signal.butter(3, criticalFreqs[0],btype='high')
		#filteredPupilSizes=scipy.signal.filtfilt(b,a,filteredPupilSizesTemp)
		#
		#frameNumbersAndPupilSizes=[[frameNumbersAndPupilSizes[index][0],filteredPupilSizes[index]] for index in range(len(filteredPupilSizes))]
		
		#z-score per session. No, let's just 0-mean: the eye tracker is always at similar distance from the eye.
		#This is not important, because the whole thing is 0-meaned again after setting the data that falls between trials to 0,
		#but it's not a problem either.
		
		#DEMEANING UITGEZET OP BASIS VAN GILLES EN TOMAS: DOE DAT ALLEMAAL LATER
		# meanPupilSize=numpy.mean([row[1] for row in frameNumbersAndPupilSizes])
		# #sigmaPupilSize=numpy.std([row[1] for row in frameNumbersAndPupilSizes])
		#
		# frameNumbersAndPupilSizes=[[frameNumbersAndPupilSizes[index][0],(frameNumbersAndPupilSizes[index][1]-meanPupilSize)] for index in range(len(frameNumbersAndPupilSizes))] #/sigmaPupilSize] for index in range(len(frameNumbersAndPupilSizes))]
		
		numpy.savetxt(outputPathFiltered+outputFileName+'_cleaned_pup_GT0920.txt',numpy.array(frameNumbersAndPupilSizes),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_non_interpolated_pupil.txt',numpy.array(frameNumbersAndPupilSizesBeforeInterpolation),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_xGaze_on_pupil_axis.txt',numpy.array(frameNumbersAndGazeXpoints),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
		numpy.savetxt(outputPathFiltered+outputFileName+'_yGaze_on_pupil_axis.txt',numpy.array(frameNumbersAndGazeYpoints),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers
			
	else:
		
		userinteraction.printFeedbackMessage("not cleaning up pupil data for "+outputFileName+" because already done")

def engbertMergenthalerV(XYseries,sPerFrame):
	#for microsaccade detection from Engbert & Mergenthaler, PNAS 2006

	vXYseries=[[(XYseries[index+2][0]+XYseries[index+1][0]-XYseries[index-1][0]-XYseries[index-2][0])/(6.*sPerFrame), \
			(XYseries[index+2][1]+XYseries[index+1][1]-XYseries[index-1][1]-XYseries[index-2][1])/(6.*sPerFrame)] for index in range(2,len(XYseries)-2)]

	return vXYseries

def getEllipseCoordinates(sigmaVXY,microsaccVelThresh):
	#for microsaccade detection from Engbert & Mergenthaler, PNAS 2006
	#these are the x-coordinates, and accompanying y-values for the bottom and top half of the ellipse.
	#I suspect this is simply for visualization purposes

	ellipsex_L=[-sigmaVXY[0]*microsaccVelThresh+counter*sigmaVXY[0]*microsaccVelThresh/100. for counter in range(201)]
	ellipsey_top_L=[numpy.sqrt(1-pow(this_x/(sigmaVXY[0]*microsaccVelThresh),2))*sigmaVXY[1]*microsaccVelThresh for this_x in ellipsex_L]
	ellipsey_bot_L=[-element for element in ellipsey_top_L]

	return([ellipsex_L,ellipsey_top_L,ellipsey_bot_L])

def getMicrosaccIndices(vXYseries,sigmaVXY,microsaccVelThresh,microsaccMinDur_frames=-1):
	#for microsaccade detection from Engbert & Mergenthaler, PNAS 2006
	#returns a list of lists, with each of these (sub)lists containing contiguous indices into vXYseries that correspond to a saccade, according to the algorithm
	#vXYseries is [speedX,speedY] using basically the same time steps as the X,Y data that it's based on, but offset slightly because v is based on more than 1 position
	#sigmaVXY is [sigma(vX),sigma(vY)], except based on medians instead of means
	#microsaccVelThresh is just that. its units are in terms of this sigma, then.
	#minimum duration that can be accepted as a saccade, in frames.
	#---
	#probably best to chop up your time series data into uninterrupted sections before applying any of this

	microssaccadeIndices=[index for index in range(len(vXYseries)) if (pow(vXYseries[index][0]/(sigmaVXY[0]*microsaccVelThresh),2)+pow(vXYseries[index][1]/(sigmaVXY[1]*microsaccVelThresh),2))>1]
	sequences=[]
	thisSequence=[]
	for index in microssaccadeIndices:
		if not(thisSequence):
			thisSequence=[index]
		elif index==thisSequence[-1]+1:
			thisSequence=thisSequence+[index]
		else:
			sequences=sequences+[thisSequence]
			thisSequence=[]
			
	if microsaccMinDur_frames>0:
		microssaccadeIndices=[thisSequence for thisSequence in sequences if len(thisSequence)>microsaccMinDur_frames]
	else:
		microssaccadeIndices=sequences
		
	return microssaccadeIndices

def pathLengthTotal(XYseries):

	pathLength=sum([numpy.sqrt(pow(XYseries[index+1][0]-XYseries[index][0],2)+pow(XYseries[index+1][1]-XYseries[index][1],2)) for index in range(len(XYseries)-1)])
	return pathLength

def detectSaccades(preprocessInfoDict,inputPath,outputPathTraces,outputPathRegressors,inputFileName,outputFileName):

	commandline.ShellCmd('mkdir '+outputPathRegressors)
	regressorFilesPresent=commandline.putFileNamesInArray(outputPathRegressors)

	if not(outputFileName+'_saccade_moments_and_other_info.txt' in regressorFilesPresent):

		userinteraction.printFeedbackMessage("detecting saccades for "+outputFileName)

		sampleData=filemanipulation.readDelimitedIntoArray(inputPath+inputFileName+'_s.asc','\t')

		sampleData=[[float(rowElement) if rowElement.strip(' .CR') else -1.0 for rowElement in row] for row in sampleData]	#if only contains spaces, dots C's and/or R's, then will evaluate to false because empty string after stripping.

		filename=inputPath+inputFileName+'_e.asc'
		f = open(filename, 'r')
		rawtext=f.read()
		f.close()

		#shift all time points relative to event defined in 'timeZeroInfo', and delete whatever comes before
		regExp=preprocessInfoDict['timeZeroInfo'][0]
		rankNumber=preprocessInfoDict['timeZeroInfo'][1]
		timeZero=int(re.findall(regExp,rawtext)[rankNumber])
		timeZeroIncBuffer=timeZero-(preprocessInfoDict['timeZeroInfo'][2])*1000.

		sampleData=[[element[0]-timeZero,element[1],element[2],element[4],element[5]] for element in sampleData if (element[0]>=timeZeroIncBuffer and not(-1.0 in element[:-1]))]		#this is binocular: 6 element besides moment; would otherwise be 3. indices 1 and 2 are x and y left; 4 and 5 are x and y right. Drop all missing data right away: saccades are dealt with elsewhere.

		timePerRowAscFile_ms=preprocessInfoDict['timePerRowAscFile_s']*1000.
		
		sampleDataContiguousSequences=[]
		oneContiguousSequence=[sampleData[0]]
		lastTime=oneContiguousSequence[-1][0]
		for sample in sampleData[1:]:
			thisTime=sample[0]
			if (thisTime-timePerRowAscFile_ms)==lastTime:
				oneContiguousSequence=oneContiguousSequence+[sample]
			else:
				if len(oneContiguousSequence)>=5:	#sequences shorter than 5 aren't useful because can't calculate speed
					sampleDataContiguousSequences=sampleDataContiguousSequences+[oneContiguousSequence]
				oneContiguousSequence=[sample]
			lastTime=thisTime	
		
		#add the final contiguous sequence
		if len(oneContiguousSequence)>=5:	#sequences shorter than 5 aren't useful because can't calculate speed
			sampleDataContiguousSequences=sampleDataContiguousSequences+[oneContiguousSequence]

		microsaccadeSummaryDataBothEyes=[]
		eyeTracesBothEyes=[]
		for LRshift in [0,2]:
			
			microsaccadeSummaryDataOneEye=[]
			eyeTracesOneEye=[]
			for oneContiguousSequence in sampleDataContiguousSequences:	
				
				eyeTracesOneEye=eyeTracesOneEye+[[element[0],element[1+LRshift],element[2+LRshift]] for element in oneContiguousSequence]
			
				vEngbertMergenthaler=engbertMergenthalerV([[element[1+LRshift],element[2+LRshift]] for element in oneContiguousSequence],preprocessInfoDict['timePerRowAscFile_s'])	#calculate speed
		
				axisShiftSpeedVsPos=int((len(oneContiguousSequence)-len(vEngbertMergenthaler))/2) #later shift the time points of detected saccades this many samples to the right to get them on the same axis as the xy data. The value is usually simply 2.
			
				medianV=[numpy.median([element[0] for element in vEngbertMergenthaler]),numpy.median([element[1] for element in vEngbertMergenthaler])]	#median x and y speed
				medBasedSigmaV=[numpy.sqrt(numpy.median([(element[0]-medianV[0])*(element[0]-medianV[0]) for element in vEngbertMergenthaler])),numpy.sqrt(numpy.median([(element[1]-medianV[1])*(element[1]-medianV[1]) for element in vEngbertMergenthaler]))]	#called median-based sigmas of X and Y: sqrt of  median squared distance to median
		
				#[ellipsex,ellipsey_top,ellipsey_bot]=getEllipseCoordinates(medBasedSigmaV,preprocessInfoDict['microsaccVelThresh'])	#These are the x-coordinates, and accompanying y-values for the bottom and top half of the ellipse, just for visualization so not used
		
				microssaccadeIndices=getMicrosaccIndices(vEngbertMergenthaler,medBasedSigmaV,preprocessInfoDict['microsaccVelThresh'],preprocessInfoDict['microsaccMinDur_ms']/(1000.*preprocessInfoDict['timePerRowAscFile_s'])) #Get the moments, in indices within vEnbertMergenthaler, of all saccades, based on velocity profile and a minimum duration
			
				for indicesOneSaccade in microssaccadeIndices:
				
					maxV=max([numpy.sqrt(pow(vEngbertMergenthaler[index][0],2)+pow(vEngbertMergenthaler[index][1],2)) for index in indicesOneSaccade])
				
					startIndexInPositionList=indicesOneSaccade[0]-axisShiftSpeedVsPos
					endIndexInPositionList=indicesOneSaccade[-1]-axisShiftSpeedVsPos
				
					startEndPos=[[oneContiguousSequence[startIndexInPositionList][1+LRshift],oneContiguousSequence[startIndexInPositionList][2+LRshift]],[oneContiguousSequence[endIndexInPositionList][1+LRshift],oneContiguousSequence[endIndexInPositionList][2+LRshift]]]
					displacementStartEnd=pathLengthTotal(startEndPos)
				
					pathLength=pathLengthTotal([[oneContiguousSequence[index-axisShiftSpeedVsPos][1+LRshift],oneContiguousSequence[index-axisShiftSpeedVsPos][2+LRshift]] for index in indicesOneSaccade])
				
					oneSaccade={'startTime_ms':oneContiguousSequence[startIndexInPositionList][0],'endTime_ms':oneContiguousSequence[endIndexInPositionList][0],'maxSpeed':maxV,'displacement':displacementStartEnd,'pathLength':pathLength}
					microsaccadeSummaryDataOneEye=microsaccadeSummaryDataOneEye+[oneSaccade]
					
			microsaccadeSummaryDataBothEyes=microsaccadeSummaryDataBothEyes+[microsaccadeSummaryDataOneEye]
			eyeTracesBothEyes=eyeTracesBothEyes+[eyeTracesOneEye]
			
		#combine into binocular saccades
		microsaccadeSummaryDataLeftEye=microsaccadeSummaryDataBothEyes[0]
		microsaccadeSummaryDataRightEye=microsaccadeSummaryDataBothEyes[1]
		
		microsaccadeSummaryDataEyesIntegrated=[]
		rightIndicesAlreadyMatched=[]
		for oneLeftSaccade in microsaccadeSummaryDataLeftEye:
			matchingRightSaccadeIndices=[index for index, element in enumerate(microsaccadeSummaryDataRightEye) if element['startTime_ms']<=oneLeftSaccade['endTime_ms'] and element['endTime_ms']>=oneLeftSaccade['startTime_ms'] and not(index in rightIndicesAlreadyMatched)]
			
			if matchingRightSaccadeIndices:	#there is at least one right-eye saccade that overlaps this left-eye one. The following code only allows one RE saccade to overlap with a given LE saccade. If there are, in fact, multiple, then that's weird, so tough nuggies.
				
				matchingRightSaccadeIndex=matchingRightSaccadeIndices[0]
				
				rightIndicesAlreadyMatched=rightIndicesAlreadyMatched+[matchingRightSaccadeIndex]
				matchingRightSaccade=microsaccadeSummaryDataRightEye[matchingRightSaccadeIndex]
			
				startTimeBinoc=min([oneLeftSaccade['startTime_ms'],matchingRightSaccade['startTime_ms']])
				endTimeBinoc=max([oneLeftSaccade['endTime_ms'],matchingRightSaccade['endTime_ms']])
				maxSpeedBinoc=max([oneLeftSaccade['maxSpeed'],matchingRightSaccade['maxSpeed']])
				displacementBinoc=numpy.average([oneLeftSaccade['displacement'],matchingRightSaccade['displacement']])
				pathLengthBinoc=numpy.average([oneLeftSaccade['pathLength'],matchingRightSaccade['pathLength']])
				ocularity=2
				
				microsaccadeSummaryDataEyesIntegrated=microsaccadeSummaryDataEyesIntegrated+[{'startTime_ms':startTimeBinoc,'endTime_ms':endTimeBinoc,'maxSpeed':maxSpeedBinoc,'displacement':displacementBinoc,'pathLength':pathLengthBinoc,'ocularity':ocularity}]
			
			else:	#there is no RE match for this LE saccade
				
				extraLeftSaccade=oneLeftSaccade
				extraLeftSaccade['ocularity']=0
				microsaccadeSummaryDataEyesIntegrated=microsaccadeSummaryDataEyesIntegrated+[extraLeftSaccade]
				
		for oneIndex, oneRightSaccade in enumerate(microsaccadeSummaryDataRightEye):
			if not(oneIndex) in rightIndicesAlreadyMatched:	#only the RE saccades that haven't been matched to a LE saccade
				extraRightSaccade=oneRightSaccade
				extraRightSaccade['ocularity']=1
				microsaccadeSummaryDataEyesIntegrated=microsaccadeSummaryDataEyesIntegrated+[extraRightSaccade]
				
		microsaccadeSummaryDataEyesIntegratedArray=[[element['startTime_ms']/1000.,element['endTime_ms']/1000.,element['maxSpeed'],element['displacement'],element['pathLength'],element['ocularity']] for element in microsaccadeSummaryDataEyesIntegrated]
		microsaccadeSummaryDataEyesIntegratedArray.sort()
		microsaccadeSummaryDataEyesIntegratedArray=numpy.array(microsaccadeSummaryDataEyesIntegratedArray)	#do this after sorting because sorting works differently on numpy arrays
		microsaccades_StandardRegressorFormat=[[(element[0]+element[1])/2.,1.] for element in microsaccadeSummaryDataEyesIntegratedArray if element[-1]==2.]	#use only binocular saccades
		
		[numpy.savetxt(outputPathTraces+outputFileName+'_cleaned_gaze_'+['L','R'][index]+'.txt',numpy.array(eyeTracesBothEyes[index]),fmt='%d\t%20.10f\t%20.10f') for index in range(2)]	#units are ms relative to start of trial
		numpy.savetxt(outputPathRegressors+outputFileName+'_saccade_moments_and_other_info.txt',microsaccadeSummaryDataEyesIntegratedArray,fmt='%20.10f\t%20.10f\t%20.10f\t%20.10f\t%20.10f\t%d')	#units are seconds relative to start of trial
		numpy.savetxt(outputPathRegressors+outputFileName+'_binocularsaccadeMomentsStandardRegressorFormat.txt',numpy.array(microsaccades_StandardRegressorFormat),fmt='%20.10f\t%20.10f')	#this one has the same format (i.e. moment, value) as the behavioral regressors will have
		
	else:

		userinteraction.printFeedbackMessage("not detecting saccades for "+outputFileName+" because already done")
		
def reviewPlots(timeCoursePath,eventsPath,figuresPath,obsPlusSessionString,timeChunkPerPlot_s):
#this function shows you what the tidied-up eye data and some event regressors look like together
#this function makes use of extremely non-generic code when it comes to how to process the pupil and saccade event data specifically. As indicated below.

	cleanedGazeFigsPresent=[element for element in commandline.putFileNamesInArray(figuresPath) if obsPlusSessionString+'_cleaned_gaze_startpoint_' in element]
	
	if not(cleanedGazeFigsPresent):
			
		userinteraction.printFeedbackMessage("saving cleaned gaze plots to file for "+obsPlusSessionString)
		
		colors=['b','g','r','c','m','y','k',(.65,.65,.65),(.35,.35,.35),(.15,.15,.15)]
		symbols=['o','x','+']
		colorSymbolCombis=[]
		for oneColor in colors:
			colorSymbolCombis=colorSymbolCombis+[[oneColor,oneSymbol] for oneSymbol in symbols]
		
		timeCourseFilesPresent=[element for element in commandline.putFileNamesInArray(timeCoursePath) if obsPlusSessionString in element and not('projectionLenDifference' in element)]
	
		organizedTimeCourseData=[]
		for timeCourseFile in timeCourseFilesPresent:
			fileName=timeCourseFile.strip('.txt')
			timeCourseData=filemanipulation.readDelimitedIntoArray(timeCoursePath+timeCourseFile,'\t')
		
			for column in range(1,len(timeCourseData[0])):
				theseData=[[element[0]/1000.,element[column]] for element in timeCourseData]
				organizedTimeCourseData=organizedTimeCourseData+[{'fileName':fileName,'columnForY':column,'data':theseData}]
		
		eventFilesPresent=[element for element in commandline.putFileNamesInArray(eventsPath) if obsPlusSessionString in element]
	
		#the following is extremely non-generic:
		organizedEventData=[]
		for eventFile in eventFilesPresent:
			fileName=eventFile[:-4]
			if 'blink_moments_and_durations' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				eventData=[re.findall('([-.0-9]*) *([-.0-9]*)',element[0])[0] for element in eventData]	#for some stupid upstream reason this is a string otherwise.
				eventData=[[float(element[0]),float(element[1])] for element in eventData]
			
				theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'blinkstart','data':theseData}]
			
				theseData=[[[element[0]+element[1],element[0]+element[1]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'blinkend','data':theseData}]
		
			elif 'saccade_moments_and_other_info' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				eventData=[[element[0],element[1],element[-1]] for element in eventData]

				for ocularity in [0,1,2]:
					theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData if element[2]==ocularity]
					organizedEventData=organizedEventData+[{'eventType':'saccadestart_'+str(ocularity),'data':theseData}]
			
				for ocularity in [0,1,2]:
					theseData=[[[element[1],element[1]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData if element[2]==ocularity]
					organizedEventData=organizedEventData+[{'eventType':'saccadend_'+str(ocularity),'data':theseData}]
		
			elif 'dirChangeReportEvent' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'dirChangeReport','data':theseData}]
			
			elif 'dirChangePhysEvent' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'dirChangePhysEvent','data':theseData}]
			
			elif 'probeEvent' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'probeEvent','data':theseData}]		
				
			elif 'probeReportEvent' in fileName:
				eventData=filemanipulation.readDelimitedIntoArray(eventsPath+eventFile,'\t')
				theseData=[[[element[0],element[0]],[0,1+len(organizedTimeCourseData)/10.]] for element in eventData]
				organizedEventData=organizedEventData+[{'eventType':'probeReportEvent','data':theseData}]
					
		firstTimePoint=organizedTimeCourseData[0]['data'][0][0]
		lastTimePoint=organizedTimeCourseData[0]['data'][-1][0]
	
		startTimePoint=firstTimePoint
		while startTimePoint<lastTimePoint:
		
			fig = pl.figure(figsize = (25,15))
			s = fig.add_subplot(1,1,1)
			s.set_title(obsPlusSessionString)
		
			endTimePoint=min([startTimePoint+timeChunkPerPlot_s,lastTimePoint])
		
			for timecourseIndex,oneTimeCourseData in enumerate(organizedTimeCourseData):
			
				theseData=[element for element in oneTimeCourseData['data'] if element[0]>=startTimePoint and element[0]<=endTimePoint]
			
				theseXdata=[element[0] for element in theseData]
				theseYdata=[element[1] for element in theseData]
				if len(theseXdata)>0:
					maxYdata=max(theseYdata)
					minYdata=min(theseYdata)
					if not(minYdata==maxYdata):
						theseScaledYdata=[(element-minYdata)/(maxYdata-minYdata)+float(timecourseIndex)/10. for element in theseYdata]
					else:
						theseScaledYdata=theseYdata
				else:
					theseScaledYdata=theseYdata
				
				pl.plot(theseXdata,theseScaledYdata, color = colors[timecourseIndex], linewidth=1.2, label='_'.join(oneTimeCourseData['fileName'].split('_')[1:])+' column '+str(oneTimeCourseData['columnForY']))
			
			for eventIndex, oneEventData in enumerate(organizedEventData):
			
				theseData=[element for element in oneEventData['data'] if element[0][0]>=startTimePoint and element[0][1]<=endTimePoint]
			
				if not ('_0' in oneEventData['eventType'] or '_1' in oneEventData['eventType']):	#I don't want monocular saccades and blinks to clutter the plots.
					[pl.plot(element[0],element[1], color = colorSymbolCombis[eventIndex][0], linewidth=.5) for element in theseData]
			
					flatList=[item for sublist in theseData for item in sublist]
					pl.scatter([flatList[index] for index in range(0,len(flatList),2)], [flatList[index] for index in range(1,len(flatList),2)], color = colorSymbolCombis[eventIndex][0], marker = colorSymbolCombis[eventIndex][1], label=oneEventData['eventType'])
			
			pl.legend()
			pl.savefig(figuresPath+obsPlusSessionString+'_cleaned_gaze_startpoint_'+str(startTimePoint)+'.pdf')
			pl.close()
			startTimePoint=startTimePoint+timeChunkPerPlot_s
	
	else:
		
		userinteraction.printFeedbackMessage("not saving cleaned gaze plots to file for "+obsPlusSessionString+' because already done')

def isolateAndNormalizePursuitComponent(timeCoursePath,outputPathZeroCrossings,eventsPath,figuresPath,obsPlusSessionString,bufferDeletedAroundSaccades_s,bufferDeletedAroundBlinks_s,bufferBeforeAfterAngleInterpolatedAroundBlinks_s,timeChunkPerAngleEstimation_s,timeChunkPerPlot_s):
	#bufferDeletedAroundBlinks_s is used to remove position information. But eventually we're working with displacement angle, and if position is frozen then that starts acting weird. In addition, it appears there's some opposite-to-OKN-slow-phase correction 
	#after each blink, so bufferBeforeAfterAngleInterpolatedAroundBlinks_s is used to interpolate the *angle* not the position after initial determination of angle has been done. Also, bufferBeforeAfterAngleInterpolatedAroundBlinks_s is a triplet: s before, s after, and s flanking interval whose average value
	#is used for interpolation. 
	
	figureScaler=1.#.5	#to quickly change figure size if external monitor isn't present.
	
	commandline.ShellCmd('mkdir '+outputPathZeroCrossings)
	
	colors=['b','g','r','c','m','y','k']
	symbols=['o','x']
	colorSymbolCombis=[]
	for oneColor in colors:
		colorSymbolCombis=colorSymbolCombis+[[oneColor,oneSymbol] for oneSymbol in symbols]
			
	eventFilesPresent=[element for element in commandline.putFileNamesInArray(eventsPath) if obsPlusSessionString in element]
	
	trialRegressorFile=[thisFile for thisFile in eventFilesPresent if 'trialOnset' in thisFile][0]
	trialEndRegressorFile=[thisFile for thisFile in eventFilesPresent if 'trialOffset' in thisFile][0]
	dirChangeReportedRegressorFiles=[thisFile for thisFile in eventFilesPresent if 'dirChangeReportEvent' in thisFile]
	dirChangePhysRegressorFiles=[thisFile for thisFile in eventFilesPresent if 'dirChangePhysEvent' in thisFile]
	
	if dirChangeReportedRegressorFiles:
		dirChangeReportedRegressorFile=dirChangeReportedRegressorFiles[0]
	else:
		dirChangeReportedRegressorFile=None
		
	if dirChangePhysRegressorFiles:
		dirChangePhysRegressorFile=dirChangePhysRegressorFiles[0]
	else:
		dirChangePhysRegressorFile=None
		
	zeroCrossingFilePresent=[element for element in commandline.putFileNamesInArray(outputPathZeroCrossings) if obsPlusSessionString in element and '_zero_crossings' in element]

	if not(zeroCrossingFilePresent):
		
		userinteraction.printFeedbackMessage("isolating smooth pursuit component non-rivalry for "+obsPlusSessionString)
		
		#get info to delineate where trials are and what the motion directions were
		trialInfo=filemanipulation.readDelimitedIntoArray(eventsPath+trialRegressorFile,'\t')
		trialEndInfo=filemanipulation.readDelimitedIntoArray(eventsPath+trialEndRegressorFile,'\t')
		perTrialData=[[trialInfo[index][0],trialEndInfo[index][0],trialInfo[index][2],trialInfo[index][3]] for index in range(len(trialInfo))]		#start time s, end time s, angle 1, angle 2
	
		#get stimulus dir change data
		if dirChangeReportedRegressorFile:
			dirChangeInfoReported=filemanipulation.readDelimitedIntoArray(eventsPath+dirChangeReportedRegressorFile,'\t')
		else:
			dirChangeInfoReported=[]
			
		if dirChangePhysRegressorFile:
			dirChangeInfoPhys=filemanipulation.readDelimitedIntoArray(eventsPath+dirChangePhysRegressorFile,'\t')
		else:
			dirChangeInfoPhys=[]
				
		#get info to get eye movement directions between saccades
		timeCourseFilesPresent=[element for element in commandline.putFileNamesInArray(timeCoursePath) if obsPlusSessionString in element]
	
		positionFiles=[thisFile for thisFile in timeCourseFilesPresent if 'cleaned_gaze' in thisFile]
		positionData=[filemanipulation.readDelimitedIntoArray(timeCoursePath+thisFile,'\t') for thisFile in positionFiles]
		
		saccadeFile=[thisFile for thisFile in eventFilesPresent if 'saccade_moments_and_other_info' in thisFile][0]
		saccadeInfo=filemanipulation.readDelimitedIntoArray(eventsPath+saccadeFile,'\t')
	
		blinkFile=[thisFile for thisFile in eventFilesPresent if 'blink_moments_and_durations' in thisFile][0]
		blinkInfo=filemanipulation.readDelimitedIntoArray(eventsPath+blinkFile,'\t')
		blinkInfo=[[float(element) for element in line[0].split(' ')  if not element==''] for line  in blinkInfo]

		periDirChangeData=[]
		preDirChangeBuffer=2.
		postDirChangeBuffer=4.
	
		for trialIndex,oneTrial in enumerate(perTrialData):
			
			print('working on trial '+str(trialIndex))
			trialStart_s=oneTrial[0]
			trialStart_ms=trialStart_s*1000.
			trialEnd_s=oneTrial[1]
			trialEnd_ms=trialEnd_s*1000.
	
			print('beware! messed around with anglesThisTrial to make it work, but not entirely sure why it works now')
			#anglesThisTrial=[oneTrial[2],oneTrial[3]]
			anglesThisTrial=[1.5707963268,4.7123889804]
		
			LEdataThisTrial=[[element[0]/1000.,element[1],element[2]] for element in positionData[0] if element[0]>=trialStart_ms and element[0]<=trialEnd_ms]	#let's immediately convert this to seconds like everything else to avoid confusion
			REdataThisTrial=[[element[0]/1000.,element[1],element[2]] for element in positionData[1] if element[0]>=trialStart_ms and element[0]<=trialEnd_ms]
			binocDataThisTrial=[[LEdataThisTrial[index][0],numpy.average([LEdataThisTrial[index][1],REdataThisTrial[index][1]]),numpy.average([LEdataThisTrial[index][2],REdataThisTrial[index][2]])] for index in range(len(LEdataThisTrial))]
	
			saccadesThisTrial=[[element[0],element[1],element[-1]] for element in saccadeInfo if element[0]<=trialEnd_s and element[1]>=trialStart_s]	#start time, end time, ocularity
	
			blinksThisTrial=[[element[0],element[0]+element[1]] for element in blinkInfo if element[0]<=trialEnd_s and (element[0]+element[1])>=trialStart_s]	#start time, end time
			maintainIndices=[index for index,value in enumerate(LEdataThisTrial) if sum([value[0]>=(thisSaccade[0]-bufferDeletedAroundSaccades_s) and value[0]<=(thisSaccade[1]+bufferDeletedAroundSaccades_s) for thisSaccade in saccadesThisTrial])==0 and sum([value[0]>=(thisBlink[0]-bufferDeletedAroundBlinks_s) and value[0]<=(thisBlink[1]+bufferDeletedAroundBlinks_s) for thisBlink in blinksThisTrial])==0]	#which indices to maintain
			
			if len(maintainIndices)>1:
				nextIndex=maintainIndices[0]
				offsetLE=[0,LEdataThisTrial[nextIndex][1],LEdataThisTrial[nextIndex][2]]	#first element of the offset will always remain 0 so does nothing, but including it in the array is easier later
				offsetRE=[0,REdataThisTrial[nextIndex][1],REdataThisTrial[nextIndex][2]]
				offsetBinoc=[0,binocDataThisTrial[nextIndex][1],binocDataThisTrial[nextIndex][2]]
	
				LEdataThisTrialCollated=[[LEdataThisTrial[nextIndex][index012]-offsetLE[index012] for index012 in range(3)]]	#equals 0, for indices 1 and 2, but written this way for clarity
				REdataThisTrialCollated=[[REdataThisTrial[nextIndex][index012]-offsetRE[index012] for index012 in range(3)]]
				binocDataThisTrialCollated=[[binocDataThisTrial[nextIndex][index012]-offsetBinoc[index012] for index012 in range(3)]]
	
				lastIndex=nextIndex
				for nextIndex in maintainIndices[1:]:
					if not(nextIndex==lastIndex+1):
						offsetLE=[0]+[offsetLE[index12]-LEdataThisTrial[lastIndex][index12]+LEdataThisTrial[nextIndex][index12] for index12 in range(1,3)]	#if a saccade has happened, then you want cut the displacement due to the saccade (i.e. LEdataThisTrialCollated[-1]-LEdataThisTrial[nextIndex] out of the trace)
						offsetRE=[0]+[offsetRE[index12]-REdataThisTrial[lastIndex][index12]+REdataThisTrial[nextIndex][index12] for index12 in range(1,3)]
						offsetBinoc=[0]+[offsetBinoc[index12]-binocDataThisTrial[lastIndex][index12]+binocDataThisTrial[nextIndex][index12] for index12 in range(1,3)]
				
					LEdataThisTrialCollated=LEdataThisTrialCollated+[[LEdataThisTrial[nextIndex][index012]-offsetLE[index012] for index012 in range(3)]]
					REdataThisTrialCollated=REdataThisTrialCollated+[[REdataThisTrial[nextIndex][index012]-offsetRE[index012] for index012 in range(3)]]
					binocDataThisTrialCollated=binocDataThisTrialCollated+[[binocDataThisTrial[nextIndex][index012]-offsetBinoc[index012] for index012 in range(3)]]
					lastIndex=nextIndex
	
				#NO. Collating blinks just like saccades now. If you don't like that, uncomment the section below, delete the line above where blinksThisTrial is calculated, and alter the line above at which maintainIndices is created 
				# #blinks are treated differently from saccades: the data points immediately surrounding blinks are deleted based on bufferDeletedAroundBlinks_s, but any net displacement during the blink is not ignored. It makes sense to do the latter only for saccades.
				# blinksThisTrial=[[element[0],element[0]+element[1]] for element in blinkInfo if element[0]<=trialEnd_s and (element[0]+element[1])>=trialStart_s]	#start time, end time
				# maintainIndicesAfterBlinks=[index for index,value in enumerate(LEdataThisTrialCollated) if sum([value[0]>=(thisBlink[0]-bufferDeletedAroundBlinks_s) and value[0]<=(thisBlink[1]+bufferDeletedAroundBlinks_s) for thisBlink in blinksThisTrial])==0]	#which indices to maintain
				# 
				# LEdataThisTrialCollated=[LEdataThisTrialCollated[oneIndex] for oneIndex in maintainIndicesAfterBlinks]
				# REdataThisTrialCollated=[REdataThisTrialCollated[oneIndex] for oneIndex in maintainIndicesAfterBlinks]
				# binocDataThisTrialCollated=[binocDataThisTrialCollated[oneIndex] for oneIndex in maintainIndicesAfterBlinks]
		
				LEanglesThisTrial=[]
				REanglesThisTrial=[]
				binocAnglesThisTrial=[]
				projectionLengthDifferenceThisTrial=[]
	
				startTimeAngleChunk=trialStart_s
	
				while startTimeAngleChunk<trialEnd_s:
		
					endTimeAngleChunk=startTimeAngleChunk+timeChunkPerAngleEstimation_s
					theseIndicesAngleChunk=[index for index,element in enumerate(LEdataThisTrialCollated) if element[0]>=startTimeAngleChunk and element[0]<=endTimeAngleChunk]
		
					if len(theseIndicesAngleChunk)>2:	#fewer and you can't fit the curves
			
						timepointsThisChunk=[LEdataThisTrialCollated[oneIndex][0] for oneIndex in theseIndicesAngleChunk]
						timepointForAngle=numpy.average(timepointsThisChunk)
		
						#fit linear curve to this little segment of the X position data, and the Y position data, separately. Then the angle is defined by the slope relation between the two.
						paramsXvsTL = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [LEdataThisTrialCollated[oneIndex][1] for oneIndex in theseIndicesAngleChunk])
						paramsYvsTL = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [LEdataThisTrialCollated[oneIndex][2] for oneIndex in theseIndicesAngleChunk])
			
						slopeXvsTL=paramsXvsTL[0][0]
						slopeYvsTL=paramsYvsTL[0][0]
			
						angleLE=numpy.pi-numpy.arctan2(slopeXvsTL,slopeYvsTL)	#pi- is to get it on the same scale as the stimulus direction data.
			
						paramsXvsTR = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [REdataThisTrialCollated[oneIndex][1] for oneIndex in theseIndicesAngleChunk])
						paramsYvsTR = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [REdataThisTrialCollated[oneIndex][2] for oneIndex in theseIndicesAngleChunk])
			
						slopeXvsTR=paramsXvsTR[0][0]
						slopeYvsTR=paramsYvsTR[0][0]
			
						angleRE=numpy.pi-numpy.arctan2(slopeXvsTR,slopeYvsTR)	#pi- is to get it on the same scale as the stimulus direction data.
			
						paramsXvsTbinoc = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [binocDataThisTrialCollated[oneIndex][1] for oneIndex in theseIndicesAngleChunk])
						paramsYvsTbinoc = scipy.optimize.curve_fit(utilities.linFunc, timepointsThisChunk, [binocDataThisTrialCollated[oneIndex][2] for oneIndex in theseIndicesAngleChunk])
			
						slopeXvsTbinoc=paramsXvsTbinoc[0][0]
						slopeYvsTbinoc=paramsYvsTbinoc[0][0]
			
						angleBinoc=numpy.pi-numpy.arctan2(slopeXvsTbinoc,slopeYvsTbinoc)	#pi- is to get it on the same scale as the stimulus direction data.
		
						LEanglesThisTrial=LEanglesThisTrial+[[timepointForAngle,angleLE]]
						REanglesThisTrial=REanglesThisTrial+[[timepointForAngle,angleRE]]
						binocAnglesThisTrial=binocAnglesThisTrial+[[timepointForAngle,angleBinoc]]
					
						#the following line makes projectionLengthDifferenceThisTrial relative to the EYES that have anglesThisTrial[0] (in which case 1) and anglesThisTrial[1] (in which case -1)
						projectionLengthDifferenceThisTrial=projectionLengthDifferenceThisTrial+[[timepointForAngle,(numpy.cos(angleBinoc-anglesThisTrial[0])-numpy.cos(angleBinoc-anglesThisTrial[1]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))]]
		
					startTimeAngleChunk=startTimeAngleChunk+(timeChunkPerAngleEstimation_s/20.)#sliding window
	
				#replace angle within certain region around blink with reasonable average of surrounding stuff
				for oneBlink in blinksThisTrial:
				
					beforeFlankerVals=[element[1] for element in projectionLengthDifferenceThisTrial if element[0]>(oneBlink[0]-bufferBeforeAfterAngleInterpolatedAroundBlinks_s[0]-bufferBeforeAfterAngleInterpolatedAroundBlinks_s[2]) and element[0]<(oneBlink[0]-bufferBeforeAfterAngleInterpolatedAroundBlinks_s[0])]
					afterFlankerVals=[element[1] for element in projectionLengthDifferenceThisTrial if element[0]>(oneBlink[1]+bufferBeforeAfterAngleInterpolatedAroundBlinks_s[1]) and element[0]<(oneBlink[1]+bufferBeforeAfterAngleInterpolatedAroundBlinks_s[1]+bufferBeforeAfterAngleInterpolatedAroundBlinks_s[2])]
				
					if beforeFlankerVals and afterFlankerVals:		#if there are any values in the flanking intervals
						averageBefore=numpy.average(beforeFlankerVals)
						averageAfter=numpy.average(afterFlankerVals)
						averageOverall=numpy.average([averageBefore,averageAfter])
					
						for projLengthIndex,projLengthData in enumerate(projectionLengthDifferenceThisTrial):
							if projLengthData[0]>(oneBlink[0]-bufferBeforeAfterAngleInterpolatedAroundBlinks_s[0]) and projLengthData[0]<(oneBlink[1]+bufferBeforeAfterAngleInterpolatedAroundBlinks_s[1]):
								projectionLengthDifferenceThisTrial[projLengthIndex]=[projectionLengthDifferenceThisTrial[projLengthIndex][0],averageOverall]
			
				numpy.savetxt(timeCoursePath+obsPlusSessionString+'_trial_'+str(trialIndex)+'_projectionLenDifference.txt',numpy.array([[element[0]*1000.,element[1]] for element in projectionLengthDifferenceThisTrial]),fmt='%d\t%20.10f')	#units are ms relative to start of trial; not frame numbers	
			
				#the following line makes dirChangeInfoNormalizedThisTrial relative to the EYES that have anglesThisTrial[0] (in which case 1) and anglesThisTrial[1] (in which case -1)
				#dirChangeInfoNormalizedThisTrial=[[thisDirChange[0],(numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[0])-numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[1]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))] for thisDirChange in dirChangeInfoReported if thisDirChange[0]<=trialEnd_s and thisDirChange[0]>=trialStart_s]	#1 and -1 for switches to first and second element in anglesThisTrial, respectively
			
				dirChangeInfoNormalizedThisTrialReported=[]
				for thisDirChange in dirChangeInfoReported:
					if thisDirChange[0]<=trialEnd_s and thisDirChange[0]>=trialStart_s:
						if thisDirChange[2]==2:
							newEntry=[thisDirChange[0],0]
						else:
							newEntry=[thisDirChange[0],(numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[1])-numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[0]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))]
						dirChangeInfoNormalizedThisTrialReported=dirChangeInfoNormalizedThisTrialReported+[newEntry]	
			
				dirChangeInfoNormalizedThisTrialPhys=[]
				for thisDirChange in dirChangeInfoPhys:
					if thisDirChange[0]<=trialEnd_s and thisDirChange[0]>=trialStart_s:
						if thisDirChange[2]==2:
							newEntry=[thisDirChange[0],0]
						else:
							newEntry=[thisDirChange[0],(numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[1])-numpy.cos(anglesThisTrial[int(thisDirChange[2])]-anglesThisTrial[0]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))]
						dirChangeInfoNormalizedThisTrialPhys=dirChangeInfoNormalizedThisTrialPhys+[newEntry]	
			
				projLengthReDirChangesThisTrialReported=[]
				for oneDirChangeInfoNorm in dirChangeInfoNormalizedThisTrialReported:
					if not oneDirChangeInfoNorm[1]==0.:
						# try:
						# 	relevantProjLengthData=[[element[0]-oneDirChangeInfoNorm[0],element[1]*oneDirChangeInfoNorm[1]] for element in projectionLengthDifferenceThisTrial if element[0]>oneDirChangeInfoNorm[0]-preDirChangeBuffer and element[0]<oneDirChangeInfoNorm[0]+postDirChangeBuffer]
						# except TypeError:
						# 	shell()
						relevantProjLengthData=[[element[0]-oneDirChangeInfoNorm[0],-element[1]*oneDirChangeInfoNorm[1]] for element in projectionLengthDifferenceThisTrial if element[0]>oneDirChangeInfoNorm[0]-preDirChangeBuffer and element[0]<oneDirChangeInfoNorm[0]+postDirChangeBuffer]
						projLengthReDirChangesThisTrialReported=projLengthReDirChangesThisTrialReported+[relevantProjLengthData]
				
				projLengthReDirChangesThisTrialPhys=[]
				for oneDirChangeInfoNorm in dirChangeInfoNormalizedThisTrialPhys:
					if not oneDirChangeInfoNorm[1]==0.:
						relevantProjLengthData=[[element[0]-oneDirChangeInfoNorm[0],element[1]*oneDirChangeInfoNorm[1]] for element in projectionLengthDifferenceThisTrial if element[0]>oneDirChangeInfoNorm[0]-preDirChangeBuffer and element[0]<oneDirChangeInfoNorm[0]+postDirChangeBuffer]
						projLengthReDirChangesThisTrialPhys=projLengthReDirChangesThisTrialPhys+[relevantProjLengthData]
			
				periDirChangeData=periDirChangeData+[{'angleDiff':numpy.mod(abs(anglesThisTrial[0]-anglesThisTrial[1]),numpy.pi+.000001),'periDataReported':projLengthReDirChangesThisTrialReported,'periDataPhys':projLengthReDirChangesThisTrialPhys}]
	
				firstTimePoint=LEdataThisTrialCollated[0][0]
				lastTimePoint=LEdataThisTrialCollated[-1][0]

				startTimePoint=firstTimePoint
				while startTimePoint<lastTimePoint:

					endTimePoint=min([startTimePoint+timeChunkPerPlot_s,lastTimePoint])
		
					fig = pl.figure(figsize = (25*figureScaler,20*figureScaler))
					s = fig.add_subplot(3,1,1)
					s.set_title(obsPlusSessionString+'\ntrial '+str(trialIndex))

					indicesIncluded=[index for index in range(len(LEdataThisTrialCollated)) if (LEdataThisTrialCollated[index][0]>=startTimePoint and LEdataThisTrialCollated[index][0]<=endTimePoint)]
					
					if not(indicesIncluded):
						
						userinteraction.printFeedbackMessage("not making plot for "+obsPlusSessionString+" interval "+str(startTimePoint)+" to "+str(endTimePoint)+" because it contains no data")
						
					else:
					
						theseLEXdata=[[LEdataThisTrialCollated[index][0], LEdataThisTrialCollated[index][1]] for index in indicesIncluded]
						theseLEYdata=[[LEdataThisTrialCollated[index][0], LEdataThisTrialCollated[index][2]] for index in indicesIncluded]
						theseREXdata=[[REdataThisTrialCollated[index][0], REdataThisTrialCollated[index][1]] for index in indicesIncluded]
						theseREYdata=[[REdataThisTrialCollated[index][0], REdataThisTrialCollated[index][2]] for index in indicesIncluded]
						
						theseBinocXdata=[[LEdataThisTrialCollated[index][0], numpy.average([LEdataThisTrialCollated[index][1],REdataThisTrialCollated[index][1]])] for index in indicesIncluded]
						theseBinocYdata=[[LEdataThisTrialCollated[index][0], numpy.average([LEdataThisTrialCollated[index][2],REdataThisTrialCollated[index][2]])] for index in indicesIncluded]
	
						organizedTimeCourseData=[theseLEXdata,theseLEYdata,theseREXdata,theseREYdata,theseBinocXdata,theseBinocYdata]
						theseTimeCourseNames=['LE_X','LE_Y','RE_X','RE_Y','binoc_X','binoc_Y']
	
						maxYvalOverall=-numpy.inf
						for timecourseIndex,oneTimeCourseData in enumerate(organizedTimeCourseData):

							theseXdata=[element[0] for element in oneTimeCourseData]
							theseYdata=[element[1] for element in oneTimeCourseData]
		
							minYdata=min(theseYdata)
							theseOffsetYdata=[(element-minYdata) for element in theseYdata]
							maxYdata=max(theseOffsetYdata)
							maxYvalOverall=max([maxYvalOverall,maxYdata])
		
							pl.plot(theseXdata,theseOffsetYdata, color = colors[timecourseIndex], linewidth=1.2, label=obsPlusSessionString+'\ntrial '+str(trialIndex)+'\n'+theseTimeCourseNames[timecourseIndex])

						saccadesStartPointsForPlot=[[[element[0]-bufferDeletedAroundSaccades_s,element[0]-bufferDeletedAroundSaccades_s],[0,maxYvalOverall+len(organizedTimeCourseData)/10.]] for element in saccadesThisTrial if element[0]<=endTimePoint and element[0]>=startTimePoint]
						saccadesEndPointsForPlot=[[[element[1]+bufferDeletedAroundSaccades_s,element[1]+bufferDeletedAroundSaccades_s],[0,maxYvalOverall+len(organizedTimeCourseData)/10.]] for element in saccadesThisTrial if element[1]<=endTimePoint and element[1]>=startTimePoint]
						blinkStartPointsForPlot=[[[element[0]-bufferDeletedAroundBlinks_s,element[0]-bufferDeletedAroundBlinks_s],[-5,maxYvalOverall+5.+len(organizedTimeCourseData)/10.]] for element in blinksThisTrial if element[0]<=endTimePoint and element[0]>=startTimePoint]
						blinksEndPointsForPlot=[[[element[1]+bufferDeletedAroundBlinks_s,element[1]+bufferDeletedAroundBlinks_s],[-5,maxYvalOverall+5.+len(organizedTimeCourseData)/10.]] for element in blinksThisTrial if element[1]<=endTimePoint and element[1]>=startTimePoint]
						dirChangePointsForPlotPhys=[[[element[0],element[0]],[0,maxYvalOverall+len(organizedTimeCourseData)/10.]] for element in dirChangeInfoPhys if element[0]<=endTimePoint and element[0]>=startTimePoint]
						dirChangePointsForPlotReported=[[[element[0],element[0]],[0,maxYvalOverall+len(organizedTimeCourseData)/10.]] for element in dirChangeInfoReported if element[0]<=endTimePoint and element[0]>=startTimePoint]
				
						organizedEventData=[saccadesStartPointsForPlot,saccadesEndPointsForPlot,blinkStartPointsForPlot,blinksEndPointsForPlot,dirChangePointsForPlotPhys,dirChangePointsForPlotReported]
						eventNames=['sacc start','sacc end','blink start','blink end','dir change phys','dir change report']

						for eventIndex, oneEventData in enumerate(organizedEventData):

							if not 'sacc' in eventNames[eventIndex]:
								[pl.plot(element[0],element[1], color = colorSymbolCombis[eventIndex][0], linewidth=.5) for element in oneEventData]

							flatList=[item for sublist in oneEventData for item in sublist]
							pl.scatter([flatList[index] for index in range(0,len(flatList),2)], [flatList[index] for index in range(1,len(flatList),2)], color = colorSymbolCombis[eventIndex][0], marker = colorSymbolCombis[eventIndex][1], label=eventNames[eventIndex])

						s.set_xlim(xmin = min([element[0] for element in theseLEXdata]), xmax = max([element[0] for element in theseLEXdata]))
						s.set_ylim(ymin = -10., ymax = 10.+maxYvalOverall+len(organizedTimeCourseData)/10.)
						pl.legend()
						#pl.show()
		
						#---and the angles
						#fig = pl.figure(figsize = (25*figureScaler,15*figureScaler))
						s = fig.add_subplot(3,1,2)
						#s.set_title(obsPlusSessionString+'\ntrial '+str(trialIndex))

						indicesIncluded=[index for index in range(len(LEanglesThisTrial)) if (LEanglesThisTrial[index][0]>=startTimePoint and LEanglesThisTrial[index][0]<=endTimePoint)]
						theseLEdata=[[LEanglesThisTrial[index][0], LEanglesThisTrial[index][1]] for index in indicesIncluded]
						theseREdata=[[REanglesThisTrial[index][0], REanglesThisTrial[index][1]] for index in indicesIncluded]
						theseBinocData=[[binocAnglesThisTrial[index][0], binocAnglesThisTrial[index][1]] for index in indicesIncluded]
	
						organizedTimeCourseData=[theseLEdata,theseREdata,theseBinocData]
						theseTimeCourseNames=['LE_angle','RE_angle','binoc_angle']
						lineThicknesses=[.8,.8,2.]
	
						for timecourseIndex,oneTimeCourseData in enumerate(organizedTimeCourseData):

							theseXdata=[element[0] for element in oneTimeCourseData]
							theseYdata=[element[1] for element in oneTimeCourseData]
		
							pl.plot(theseXdata,theseYdata, color = colors[timecourseIndex], linewidth=lineThicknesses[timecourseIndex], label=obsPlusSessionString+'\ntrial '+str(trialIndex)+'\n'+theseTimeCourseNames[timecourseIndex])

						saccadesStartPointsForPlot=[[[element[0],element[0]],[0,numpy.pi*2]] for element in saccadesThisTrial if element[0]<=endTimePoint and element[0]>=startTimePoint]
						saccadesEndPointsForPlot=[[[element[1],element[1]],[0,numpy.pi*2]] for element in saccadesThisTrial if element[1]<=endTimePoint and element[1]>=startTimePoint]
						#dirChangePointsForPlot=[[[element[0],element[0]],[0,numpy.pi*2]] for element in dirChangeInfo if element[0]<=endTimePoint and element[0]>=startTimePoint]
				
						dirMarkersForPlotPhys=[]
						for myIndex in range(len(dirChangeInfoPhys)-1):
							if (dirChangeInfoPhys[myIndex][0]<=endTimePoint and dirChangeInfoPhys[myIndex][0]>=startTimePoint) or (dirChangeInfoPhys[myIndex+1][0]<=endTimePoint and dirChangeInfoPhys[myIndex+1][0]>=startTimePoint):
								if int(dirChangeInfoPhys[myIndex][2])==2:
									theAngle=(anglesThisTrial[0]+anglesThisTrial[1])/2.
								else:
									theAngle=anglesThisTrial[int(abs(1.-dirChangeInfoPhys[myIndex][2]))]
								dirMarkersForPlotPhys=dirMarkersForPlotPhys+[[[dirChangeInfoPhys[myIndex][0],dirChangeInfoPhys[myIndex+1][0]],[theAngle,theAngle]]]
						
						#dirMarkersForPlotPhys=[[[dirChangeInfoPhys[myIndex][0],dirChangeInfoPhys[myIndex+1][0]],[anglesThisTrial[int(dirChangeInfoPhys[myIndex][2])],anglesThisTrial[int(dirChangeInfoPhys[myIndex][2])]]] for myIndex in range(len(dirChangeInfoPhys)-1) if (dirChangeInfoPhys[myIndex][0]<=endTimePoint and dirChangeInfoPhys[myIndex][0]>=startTimePoint)]
				
						dirMarkersForPlotReported=[]
						for myIndex in range(len(dirChangeInfoReported)-1):
							if (dirChangeInfoReported[myIndex][0]<=endTimePoint and dirChangeInfoReported[myIndex][0]>=startTimePoint) or (dirChangeInfoReported[myIndex+1][0]<=endTimePoint and dirChangeInfoReported[myIndex+1][0]>=startTimePoint):
								if int(dirChangeInfoReported[myIndex][2])==2:
									theAngle=(anglesThisTrial[0]+anglesThisTrial[1])/2.
								else:
									theAngle=anglesThisTrial[int(dirChangeInfoReported[myIndex][2])]
								dirMarkersForPlotReported=dirMarkersForPlotReported+[[[dirChangeInfoReported[myIndex][0],dirChangeInfoReported[myIndex+1][0]],[theAngle,theAngle]]]
				
						#dirMarkersForPlotReported=[[[dirChangeInfoReported[myIndex][0],dirChangeInfoReported[myIndex+1][0]],[anglesThisTrial[int(dirChangeInfoReported[myIndex][2])],anglesThisTrial[int(dirChangeInfoReported[myIndex][2])]]] for myIndex in range(len(dirChangeInfoReported)-1) if (dirChangeInfoReported[myIndex][0]<=endTimePoint and dirChangeInfoReported[myIndex][0]>=startTimePoint)]
					
						organizedEventData=[saccadesStartPointsForPlot,saccadesEndPointsForPlot,dirMarkersForPlotPhys,dirMarkersForPlotReported]
						eventNames=['sacc start','sacc end','stim dir phys','stim dir rep']
		
						for eventIndex, oneEventData in enumerate(organizedEventData):

							if not 'sacc' in eventNames[eventIndex]:
								[pl.plot(element[0],element[1], color = colorSymbolCombis[eventIndex][0], linewidth=.5, dashes=[4, 2]) for element in oneEventData]

							flatList=[item for sublist in oneEventData for item in sublist]
							pl.scatter([flatList[index] for index in range(0,len(flatList),2)], [flatList[index] for index in range(1,len(flatList),2)], color = colorSymbolCombis[eventIndex][0], marker = colorSymbolCombis[eventIndex][1], label=eventNames[eventIndex])

						#[pl.plot([startTimePoint,endTimePoint],[oneStimAngleThisTrial,oneStimAngleThisTrial], color = 'k', linewidth=.5) for oneStimAngleThisTrial in anglesThisTrial]
		
						s.set_ylim(ymin = -.1, ymax = .1+2.*numpy.pi)
						s.set_yticks([0,numpy.pi,numpy.pi*2.])
						try:
							s.set_xlim(xmin = min([element[0] for element in theseLEdata]), xmax = max([element[0] for element in theseLEdata]))
						except ValueError:
							print('yoyoyo')
							shell()
		
						pl.legend()
						#pl.show()
						#---
		
						#---and the projection length difference
			
						#fig = pl.figure(figsize = (25*figureScaler,15*figureScaler))
						s = fig.add_subplot(3,1,3)
						#s.set_title(obsPlusSessionString+'\ntrial '+str(trialIndex))

						indicesIncluded=[index for index in range(len(projectionLengthDifferenceThisTrial)) if (projectionLengthDifferenceThisTrial[index][0]>=startTimePoint and projectionLengthDifferenceThisTrial[index][0]<=endTimePoint)]
						theseData=[[projectionLengthDifferenceThisTrial[index][0], projectionLengthDifferenceThisTrial[index][1]] for index in indicesIncluded]
				
						#organizedTimeCourseData=[theseLEdata,theseREdata,theseBinocData]
						thisName='projection length difference'
	
						theseXdata=[element[0] for element in theseData]
						theseYdata=[element[1] for element in theseData]
		
						pl.plot(theseXdata,theseYdata, color = 'r', linewidth=2., label=obsPlusSessionString+'\ntrial '+str(trialIndex)+'\n'+thisName)

						#dirChangePointsForPlot=[[[element[0],element[0]],[-1.5,1.5]] for element in dirChangeInfo if element[0]<=endTimePoint and element[0]>=startTimePoint]
				
						#the next line makes dirMarkersForPlot relative to EYES that have anglesThisTrial[0] (in which case 1) and anglesThisTrial[1] (in which case -1)
						dirMarkersForPlotReported=[]
						for myIndex in range(len(dirChangeInfoReported)-1):
							if (dirChangeInfoReported[myIndex][0]<=endTimePoint and dirChangeInfoReported[myIndex][0]>=startTimePoint) or (dirChangeInfoReported[myIndex+1][0]<=endTimePoint and dirChangeInfoReported[myIndex+1][0]>=startTimePoint):
								if int(dirChangeInfoReported[myIndex][2])==2:
									newYval=0.
								else:
									newYval=(numpy.cos(anglesThisTrial[int(dirChangeInfoReported[myIndex][2])]-anglesThisTrial[0])-numpy.cos(anglesThisTrial[int(dirChangeInfoReported[myIndex][2])]-anglesThisTrial[1]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))
							
								newEntry=[[dirChangeInfoReported[myIndex][0],dirChangeInfoReported[myIndex+1][0]],[newYval,newYval]]
								dirMarkersForPlotReported=dirMarkersForPlotReported+[newEntry]
						
						dirMarkersForPlotPhys=[]
						for myIndex in range(len(dirChangeInfoPhys)-1):
							if (dirChangeInfoPhys[myIndex][0]<=endTimePoint and dirChangeInfoPhys[myIndex][0]>=startTimePoint) or (dirChangeInfoPhys[myIndex+1][0]<=endTimePoint and dirChangeInfoPhys[myIndex+1][0]>=startTimePoint):
								if int(dirChangeInfoPhys[myIndex][2])==2:
									newYval=0.
								else:
									newYval=(numpy.cos(anglesThisTrial[int(abs(1.-dirChangeInfoPhys[myIndex][2]))]-anglesThisTrial[0])-numpy.cos(anglesThisTrial[int(abs(1.-dirChangeInfoPhys[myIndex][2]))]-anglesThisTrial[1]))/(1.-numpy.cos(anglesThisTrial[0]-anglesThisTrial[1]))

								newEntry=[[dirChangeInfoPhys[myIndex][0],dirChangeInfoPhys[myIndex+1][0]],[newYval,newYval]]
								dirMarkersForPlotPhys=dirMarkersForPlotPhys+[newEntry]
				
						organizedEventData=[dirMarkersForPlotPhys,dirMarkersForPlotReported]
						eventNames=['stim dir phys (normalized)','stim dir rep (normalized)']
		
						for eventIndex, oneEventData in enumerate(organizedEventData):

							[pl.plot(element[0],element[1], color = colorSymbolCombis[eventIndex][0], linewidth=.5, dashes=[5, 2]) for element in oneEventData]

							flatList=[item for sublist in oneEventData for item in sublist]
							pl.scatter([flatList[index] for index in range(0,len(flatList),2)], [flatList[index] for index in range(1,len(flatList),2)], color = colorSymbolCombis[eventIndex][0], marker = colorSymbolCombis[eventIndex][1], label=eventNames[eventIndex])

						s.set_ylim(ymin = -1.50, ymax = 1.5)
						s.set_xlim(xmin = min([element[0] for element in theseData]), xmax = max([element[0] for element in theseData]))
						pl.legend()
				
						pl.savefig(figuresPath+obsPlusSessionString+'_isolated_pursuit_startpoint_'+str(startTimePoint)+'.pdf')
						pl.close()
				
					#---
		
					startTimePoint=startTimePoint+timeChunkPerPlot_s
			else:
				
				userinteraction.printFeedbackMessage("No usable data at all for "+obsPlusSessionString+", trial "+str(trialIndex)+"!")
				
				
		dataBinnedAllAnglesReported=[]
		dataBinnedAllAnglesPhys=[]
		periPlotWinSize_s=.2
		uniqueAngles=list(set([round(element['angleDiff']*1000.)/1000. for element in periDirChangeData]).union())
		for oneAngle in uniqueAngles:
			periDirChangeDataThisAngleReported=[]
			periDirChangeDataThisAnglePhys=[]
			for onePeriData in periDirChangeData:
				if round(onePeriData['angleDiff']*1000.)/1000.==oneAngle:
					periDirChangeDataThisAngleReported=periDirChangeDataThisAngleReported+onePeriData['periDataReported']
					periDirChangeDataThisAnglePhys=periDirChangeDataThisAnglePhys+onePeriData['periDataPhys']
					
			periDirChangeDataThisAngleReported=[element for element in periDirChangeDataThisAngleReported if element]	#remove empty ones.
			periDirChangeDataThisAnglePhys=[element for element in periDirChangeDataThisAnglePhys if element]	#remove empty ones.
			
			dataBinnedRep=[]
			dataBinnedPhys=[]
			winStart_s=-preDirChangeBuffer
			winEnd_s=winStart_s+periPlotWinSize_s
			while winEnd_s<=postDirChangeBuffer:
				dataAveragedPerEventThisBinRep=[]
				dataAveragedPerEventThisBinPhys=[]
				winEnd_s=winStart_s+periPlotWinSize_s
				
				if periDirChangeDataThisAngleReported:
					
					for onePeriDirSequence in periDirChangeDataThisAngleReported:
						dataOneEventThisBin=[element for element in onePeriDirSequence if element[0]>=winStart_s and element[0]<=winEnd_s]
						if len(dataOneEventThisBin)>1.:
							dataAveragedPerEventThisBinRep=dataAveragedPerEventThisBinRep+[[numpy.average([element[0] for element in dataOneEventThisBin]),numpy.average([element[1] for element in dataOneEventThisBin])]]
				
					dataBinnedRep=dataBinnedRep+[[numpy.average([element[0] for element in dataAveragedPerEventThisBinRep]),numpy.average([element[1] for element in dataAveragedPerEventThisBinRep]),numpy.std([element[1] for element in dataAveragedPerEventThisBinRep])]] #/numpy.sqrt(len(dataAveragedPerEventThisBin))
					
				if periDirChangeDataThisAnglePhys:	
					
					for onePeriDirSequence in periDirChangeDataThisAnglePhys:
						dataOneEventThisBin=[element for element in onePeriDirSequence if element[0]>=winStart_s and element[0]<=winEnd_s]
						if len(dataOneEventThisBin)>1.:
							dataAveragedPerEventThisBinPhys=dataAveragedPerEventThisBinPhys+[[numpy.average([element[0] for element in dataOneEventThisBin]),numpy.average([element[1] for element in dataOneEventThisBin])]]
	
					dataBinnedPhys=dataBinnedPhys+[[numpy.average([element[0] for element in dataAveragedPerEventThisBinPhys]),numpy.average([element[1] for element in dataAveragedPerEventThisBinPhys]),numpy.std([element[1] for element in dataAveragedPerEventThisBinPhys])]] #/numpy.sqrt(len(dataAveragedPerEventThisBin))
				
				winStart_s=winStart_s+periPlotWinSize_s/10.
		
			if dataBinnedRep:
				dataBinnedAllAnglesReported=dataBinnedAllAnglesReported+[dataBinnedRep]	
				
			if dataBinnedPhys:
				dataBinnedAllAnglesPhys=dataBinnedAllAnglesPhys+[dataBinnedPhys]	
	
		#lastSubZeroIndices=[[index for index,element in enumerate(dataBinnedOneAngle) if element[1]<0][-1] for dataBinnedOneAngle in dataBinnedAllAngles]
		#approxZeroCrossingTimes=[(dataBinnedOneAngle[lastSubZeroIndices[index]][0]*abs(dataBinnedOneAngle[lastSubZeroIndices[index]+1][1])+dataBinnedOneAngle[lastSubZeroIndices[index]+1][0]*abs(dataBinnedOneAngle[lastSubZeroIndices[index]][1]))/(abs(dataBinnedOneAngle[lastSubZeroIndices[index]][1])+abs(dataBinnedOneAngle[lastSubZeroIndices[index]+1][1])) for index, dataBinnedOneAngle in enumerate(dataBinnedAllAngles)]
	
		colors=['r','g','b']
		
		fig = pl.figure(figsize = (15,15))
		s = fig.add_subplot(1,1,1)
		s.set_title(obsPlusSessionString+'\nRGB '+' ,'.join([str(element) for element in uniqueAngles])+'\nSolid/dashed: physical/reported switch')
	
		pl.plot([-preDirChangeBuffer,postDirChangeBuffer],[0,0],color='k')
		for angleIndex,dataBinnedOneAngle in enumerate(dataBinnedAllAnglesPhys):
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1] for element in dataBinnedOneAngle],linewidth=1.2,color=colors[angleIndex])
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1]+element[2] for element in dataBinnedOneAngle],linewidth=.4,color=colors[angleIndex])
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1]-element[2] for element in dataBinnedOneAngle],linewidth=.4,color=colors[angleIndex])

		for angleIndex,dataBinnedOneAngle in enumerate(dataBinnedAllAnglesReported):
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1] for element in dataBinnedOneAngle],linewidth=1.2,color=colors[angleIndex], dashes=[6, 2])
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1]+element[2] for element in dataBinnedOneAngle],linewidth=.4,color=colors[angleIndex], dashes=[6, 2])
			pl.plot([element[0] for element in dataBinnedOneAngle], [element[1]-element[2] for element in dataBinnedOneAngle],linewidth=.4,color=colors[angleIndex], dashes=[6, 2])
			#pl.plot([approxZeroCrossingTimes[angleIndex],approxZeroCrossingTimes[angleIndex]],[-1,1],color=colors[angleIndex])
	
		s.set_ylim(ymin =-1.1, ymax = 1.1)
		s.set_xlim(xmin = -preDirChangeBuffer, xmax = postDirChangeBuffer)
		
		#pl.show()
		pl.savefig(figuresPath+obsPlusSessionString+'_isolated_pursuit_overall.pdf')
		pl.close()
		
		numpy.savetxt(outputPathZeroCrossings+obsPlusSessionString+'_zero_crossings.txt',numpy.array([[0,0] for index in range(len(uniqueAngles))]),fmt='%20.10f\t%20.10f')	#save bogus file because code checks for presence
		
		with open(outputPathZeroCrossings+obsPlusSessionString+'_periSwitchDataReported', 'w') as f:
		    pickle.dump(dataBinnedAllAnglesReported, f)
		
		with open(outputPathZeroCrossings+obsPlusSessionString+'_periSwitchDataPhysical', 'w') as f:
		    pickle.dump(dataBinnedAllAnglesPhys, f)
		
	else:
		
		userinteraction.printFeedbackMessage("not isolating smooth pursuit component for "+obsPlusSessionString+" because already done")
	
def createMatrixRowsDeconvolution(pupilDataPath,pupilDataFilenameComponent,regressorDataPath,matrixOutputPath,observerSessionString,deconvolutionDict):
	#creates matrix rows: the time course, as well as the regressors. For the latter, expects that files in regressorDataPath contain time in the first column and the regressor value accompanying that time in the second
	outputFolderName=observerSessionString+'_'+deconvolutionDict['analysisName']
	
	commandline.ShellCmd('mkdir '+matrixOutputPath)
	
	allFilesAndFoldersInMatrixOutputPath=commandline.putFileNamesInArray(matrixOutputPath)
	
	if not outputFolderName+'_finalized' in allFilesAndFoldersInMatrixOutputPath:
		
		userinteraction.printFeedbackMessage("creating matrix for "+observerSessionString+", analysis "+deconvolutionDict['analysisName'])
	
		commandline.ShellCmd('mkdir '+matrixOutputPath+outputFolderName)
		
		pupilDataFile=[fileName for fileName in commandline.putFileNamesInArray(pupilDataPath) if observerSessionString in fileName and pupilDataFilenameComponent in fileName][0]
		pupilData=filemanipulation.readDelimitedIntoArray(pupilDataPath+pupilDataFile,'\t')
		
		firstTimePoint_ms=int(pupilData[0][0])
		lastTimePoint_ms=int(pupilData[-1][0])
		
		timeStepPupilData_ms=int(deconvolutionDict['timePerRowAscFile_s']*1000)
		allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms=range(firstTimePoint_ms,lastTimePoint_ms,timeStepPupilData_ms)	#at this point not relevant yet whether any data are missing. That will be taken care of later.
		
		stepSizeDeconvolution_ms=deconvolutionDict['stepSizeDeconvolution_s']*1000.
		allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms=range(firstTimePoint_ms,lastTimePoint_ms,int(stepSizeDeconvolution_ms))
		timeIntervalDeconvolution_ms=[element*1000 for element in deconvolutionDict['timeIntervalDeconvolution_s']]	#these are both the start and end point of the interval
		
		matrixRowFilesPresent=commandline.putFileNamesInArray(matrixOutputPath+outputFolderName)
		if not 'timeVaryingSignal.txt' in matrixRowFilesPresent:
			
			userinteraction.printFeedbackMessage("about to downsample pupil timecourse. This takes a while and might create 'Mean of empty slice' warnings. Those are fine.")
			
			#The below nested loops do the following: they calculate the average pupil signal per bin within the downsampled timecourse used for deconvolution.
			#The nexted loops are equivalent to this:
			#allPupilValuesDeconvolution=[numpy.average([thisPupilData[1] for thisPupilData in pupilData if (thisPupilData[0]>=allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex] \
			#and thisPupilData[0]<allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex+1])]) for leftBinBorderIndex in range(len(allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms)-1)]
			#But make use of the fact that time points are consecutive so you can pre-select which part of the array, instead of the entire one, to search.
			allPupilValuesDeconvolution=[]
			startSearchIndexPupilData=0
			for leftBinBorderIndex in range(len(allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms)-1):
				reachedTimepointYet=False
				pupilValuesToBeAveragedForThisBin=[]
				for thisPupilIndex in range(startSearchIndexPupilData,len(pupilData)):
					thisTimePoint=pupilData[thisPupilIndex][0]
					thisPupilValue=pupilData[thisPupilIndex][1]
					if (thisTimePoint>=allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex] and thisTimePoint<allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex+1]):
						reachedTimepointYet=True
						pupilValuesToBeAveragedForThisBin=pupilValuesToBeAveragedForThisBin+[thisPupilValue]
					else:
						if reachedTimepointYet:	#i.e. you've encountered the section of time already, and are now moving out of it
							startSearchIndexPupilData=thisPupilIndex-1	#For the next loop, don't search any of the hiResElements to the left of where we are currently. Adding the -1 just to be sure. I'm fairly certain it's okay without it, too.
							allPupilValuesDeconvolution=allPupilValuesDeconvolution+[numpy.average(pupilValuesToBeAveragedForThisBin)]
							break	#out of the for loop, because we're not going to find any further matches for this leftBinBorderIndex
				if not(reachedTimepointYet):	#if, for this bin in the downsampled data, we never encountered any data in the pupil timecourse that falls into this window, then we need to mark the bin for deletion later
					if startSearchIndexPupilData==len(pupilData):
						userinteraction.printFeedbackMessage("not entirely sure what happens in this case. you're probably erroneously going to try to delete a trailing bin.")	#this shouldn't happen but let's alert user to bug when it does
					else:
						allPupilValuesDeconvolution=allPupilValuesDeconvolution+[numpy.average([])]
			
			deleteIndicesDeconvolutionTimecourse=[index for index, value in enumerate(allPupilValuesDeconvolution) if numpy.isnan(value)]
			allPupilValuesDeconvolution=[value for index, value in enumerate(allPupilValuesDeconvolution) if not index in deleteIndicesDeconvolutionTimecourse]
			
			#The below nested loops do the following: they calculate which indices into an uninterruped timecourse at the high sampling rate of the tracker, would go into each bin within the downsampled timecourse used for deconvolution.
			#The nexted loops are equivalent to this:
			#indicesIntoHighSampRatePerElementOfLowSampRate=[[index for index,hiResElement in enumerate(allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms) if \
			#(hiResElement>=allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex] and hiResElement<allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex+1])] \
			#for leftBinBorderIndex in range(len(allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms)-1)]
			#But makes use of the fact that time points are consecutive so you can pre-select which part of the array, instead of the entire one, to search.
			indicesIntoHighSampRatePerElementOfLowSampRate=[]
			startSearchIndexHiRes=0
			for leftBinBorderIndex in range(len(allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms)-1):
				reachedTimepointYet=False
				indicesIntoHiresThatPointToThisLoResItem=[]
				for hiResIndex in range(startSearchIndexHiRes,len(allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms)):
					hiResElement=allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms[hiResIndex]
					if (hiResElement>=allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex] and hiResElement<allBinBordersBetweenStartAndEndDeconvolutionSamplingRate_ms[leftBinBorderIndex+1]):
						reachedTimepointYet=True
						indicesIntoHiresThatPointToThisLoResItem=indicesIntoHiresThatPointToThisLoResItem+[hiResIndex]
					else:
						if reachedTimepointYet:	#i.e. you've encountered the section of time already, and are now moving out of it
							startSearchIndexHiRes=hiResIndex-1	#For the next loop, don't search any of the hiResElements to the left of where we are currently. Adding the -1 just to be sure. I'm fairly certain it's okay without it, too.
							indicesIntoHighSampRatePerElementOfLowSampRate=indicesIntoHighSampRatePerElementOfLowSampRate+[indicesIntoHiresThatPointToThisLoResItem]
							break	#out of the for loop, because we're not going to find any further matches for this leftBinBorderIndex
			
			numpy.savetxt(matrixOutputPath+outputFolderName+'/timeVaryingSignal.txt',allPupilValuesDeconvolution,fmt='%20.10f')
			numpy.savetxt(matrixOutputPath+outputFolderName+'/_deleteIndices.txt',deleteIndicesDeconvolutionTimecourse,fmt='%20.10f')
			with open(matrixOutputPath+outputFolderName+'/_hiResLoResMapping', 'w') as f:
			    pickle.dump(indicesIntoHighSampRatePerElementOfLowSampRate, f)
			
			userinteraction.printFeedbackMessage("done downsampling pupil timecourse.")
		else:
			
			userinteraction.printFeedbackMessage("time course file already present. loading relevant data from file.")
			deleteIndicesDeconvolutionTimecourse=[int(element[0]) for element in filemanipulation.readDelimitedIntoArray(matrixOutputPath+outputFolderName+'/_deleteIndices.txt','\t')]
			with open(matrixOutputPath+outputFolderName+'/_hiResLoResMapping', 'r') as f:
			    indicesIntoHighSampRatePerElementOfLowSampRate=pickle.load(f)
		
		numHiResSamplePerHalfDecoStepsize=int((stepSizeDeconvolution_ms/2.)/timeStepPupilData_ms)+2	#build in a buffer of 2, to accommodate for rounding stuff I'm overlooking.
		
		for regressorName in deconvolutionDict['regressorNames']:
		
			theseRegressorData=filemanipulation.readDelimitedIntoArray(regressorDataPath+observerSessionString+'_'+regressorName+'.txt','\t')
			
			try:
				eventTimes_ms=[element[0]*1000. for element in theseRegressorData]
				accompanyingRegressorValues=[element[1] for element in theseRegressorData]
			except TypeError:
				theseRegressorData=[row[0].split(' ') for row in theseRegressorData]
				theseRegressorData=[[item for item in row if not item==''] for row in theseRegressorData]
				eventTimes_ms=[float(element[0])*1000. for element in theseRegressorData]
				accompanyingRegressorValues=[float(element[1]) for element in theseRegressorData]
		
			thisTimeWithinDeconvolutionWindow=timeIntervalDeconvolution_ms[0]			#These are the various timepoints of the deconvolution curve
		
			userinteraction.printFeedbackMessage('starting matrix rows for regressor named '+regressorName)
			
			while thisTimeWithinDeconvolutionWindow<=timeIntervalDeconvolution_ms[1]:
				
				if not regressorName+'_'+str(thisTimeWithinDeconvolutionWindow/1000.)+'.txt' in matrixRowFilesPresent:
			
					matrixRowPupilDataSamplingRate=numpy.zeros(len(allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms))
			
					theseEventTimes=[event+thisTimeWithinDeconvolutionWindow for event in eventTimes_ms]
			
					#----check if assumption that theseEventTimes are in order is correct---
					testThing=theseEventTimes[:]
					testThing.sort()
					if not testThing==theseEventTimes:
						userinteraction.printFeedbackMessage("THE ASSUMPTION THAT THE EVENTS ARE SORTED TEMPORALLY IS NOT MET!!! THIS WILL CAUSE ERRONEOUS RESULTS!!! DO SOMETHING!!!.")
					#-----------------------------------------------------------------------
			
					startSearchIndexHiRes=0	#again writing a loop that takes into account the fact that theseEventTimes are ordered.
					for eventIndex,eventTime in enumerate(theseEventTimes):
						eventValue=accompanyingRegressorValues[eventIndex]
				
						reachedTimepointYet=False
						for timeStepIndex in range(startSearchIndexHiRes,len(allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms)):
							timeValue=allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms[timeStepIndex]
							if timeValue>=(eventTime-stepSizeDeconvolution_ms/2.) and timeValue<=(eventTime+stepSizeDeconvolution_ms/2.):
								reachedTimepointYet=True
								matrixRowPupilDataSamplingRate[timeStepIndex]=matrixRowPupilDataSamplingRate[timeStepIndex]+eventValue	#because you're adding here, not replacing, multiple closely spaced events of the same kind can cumulatively influence the value in a bin
							else:
								if reachedTimepointYet:	#i.e. you've encountered the section of time already, and are now moving out of it
									startSearchIndexHiRes=max(timeStepIndex-numHiResSamplePerHalfDecoStepsize,0)	#For the next loop, don't search any of the allTimePointsBetweenStartAndEndPupilDataSamplingRate_ms to the left of where we are currently, but keep in mind the stepSizeDeconvolution_ms width.
									break	#out of the for loop, because we're not going to find any further matches for this eventIndex
							
					matrixRowDeconvolutionSamplingRate=[numpy.average([matrixRowPupilDataSamplingRate[thisIndex] for thisIndex in indicesOneLowResEntry]) for indicesOneLowResEntry in indicesIntoHighSampRatePerElementOfLowSampRate]
					matrixRowDeconvolutionSamplingRate=[element for index,element in enumerate(matrixRowDeconvolutionSamplingRate) if not index in deleteIndicesDeconvolutionTimecourse]
			
					numpy.savetxt(matrixOutputPath+outputFolderName+'/'+regressorName+'_'+str(thisTimeWithinDeconvolutionWindow/1000.)+'.txt',matrixRowDeconvolutionSamplingRate,fmt='%20.10f')
					
				else:
				
					userinteraction.printFeedbackMessage('matrix row for regressor named '+regressorName+' and deconvolution time point '+str(thisTimeWithinDeconvolutionWindow/1000.)+' is already present. Not creating a new one')
				
				thisTimeWithinDeconvolutionWindow=thisTimeWithinDeconvolutionWindow+stepSizeDeconvolution_ms
				
		commandline.ShellCmd('mv '+matrixOutputPath+observerSessionString+'_'+deconvolutionDict['analysisName']+' '+matrixOutputPath+observerSessionString+'_'+deconvolutionDict['analysisName']+'_finalized')
		
	else:
		
		userinteraction.printFeedbackMessage("not creating GLM matrix for "+observerSessionString+", analysis "+deconvolutionDict['analysisName']+" because already done")

def runGLMbasedOnStoredDesignMatrix(designMatrixFolder,outputFolder,obsPlusSessionString,GLMexecutionDict,useRidgeRegression,showPlots):
	
	#concatenatedTimeCourse should be a 1 by [number of timepoints] numpy.matrix()
	#designMatrix should be a [number of timepoints] by [number of regressors] numpy.matrix()
	#use the .shape property of numpy matrices to check.
	#
	#any file names in the matrix folder that have the format string+'_'+number+'.txt' are assumed to be part of a deconvolution, where the number is the timecourse,
	#this affects the way the beta weights are stored.
	#File names that aren't part of deconvolution should not have that format. They are, however, expected to also end in '.txt'.
	
	figureScaler=1.#.5	#to quickly change figure size if external monitor isn't present.
	
	regressorsUsed=GLMexecutionDict['regressorNames']
	analysisName=GLMexecutionDict['analysisName']
	
	theseCandidateDesignMatrixFiles=commandline.putFileNamesInArray(designMatrixFolder)
	
	if not 'timeVaryingSignal.txt' in theseCandidateDesignMatrixFiles:
		userinteraction.printFeedbackMessage('no time-varying signal file found in '+designMatrixFolder+'. Not continuing')
	else:
		concatenatedTimeCourseArray=[element[0] for element in filemanipulation.readDelimitedIntoArray(designMatrixFolder+'timeVaryingSignal.txt','\t')]
		concatenatedTimeCourse=numpy.matrix(concatenatedTimeCourseArray)
	
	theseDesignMatrixFilesUsed=[oneFile for oneFile in theseCandidateDesignMatrixFiles if not(oneFile=='timeVaryingSignal.txt') and not(oneFile[0]=='_')] #An underscore in the first position of the file name is used to indicate helper filenames, such as the one that stores the mapping between the original temporal resolution and the downsampled timecourse. Those files are used by createMatrixRowsDeconvolution
		
	designMatrixArray=[[element[0] for element in filemanipulation.readDelimitedIntoArray(designMatrixFolder+thisFileName,'\t')] for thisFileName in theseDesignMatrixFilesUsed]
	
	includedRegressorIndices=[]
	designMatrixArrayWithoutEventlessRegressors=[]
	for rowIndex,designMatrixRow in enumerate(designMatrixArray):
		if len(list(set(designMatrixRow).union()))>1:	#if there's only one value in the regressor, then it's pointless to include it.
			designMatrixArrayWithoutEventlessRegressors=designMatrixArrayWithoutEventlessRegressors+[designMatrixRow]
			includedRegressorIndices=includedRegressorIndices+[rowIndex]
			
	theseDesignMatrixFilesUsedAfterRemovingEmptyOnes=[theseDesignMatrixFilesUsed[oneIndex] for oneIndex in includedRegressorIndices]	
	
	designMatrixArray=designMatrixArrayWithoutEventlessRegressors
	designMatrix=numpy.matrix(designMatrixArray).T
	
	if useRidgeRegression==False:
		
		betaWeights=((designMatrix.T*designMatrix).I*designMatrix.T)*concatenatedTimeCourse.T
		betaWeightsAsArray=numpy.squeeze(numpy.asarray(betaWeights))
	
	else:
	
		designMatrixArray=numpy.swapaxes(designMatrixArray,0,1)
		clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],fit_intercept=False)
		clf.fit(designMatrixArray,concatenatedTimeCourseArray)
		betaWeightsAsArray=clf.coef_
		betaWeights=numpy.matrix(betaWeightsAsArray).T
		
	resultsDeconvolution=[]
	resultsNonDeconvolution=[]
	
	for fileIndex,fileName in enumerate(theseDesignMatrixFilesUsedAfterRemovingEmptyOnes):
		
		if fileName.split('_')[0] in regressorsUsed:
			try:
				splitName=fileName.split('_')
				if not(len(splitName)==2):	#i.e. a string describing the regressor and the number string + '.txt'
					raise ValueError	#ValueError gets raised if the putative number string is not convertible to a number, so might as well raise the same type of error if other parts of the file name format don't support the idea that it's for deconvolution
				else:
					justNumberString=splitName[1][:-4] #remove the .txt
					timeIndicator=float(justNumberString)	#this is what throws a real ValueError if the first part is not a string indicating a number
					regressorName=splitName[0]
					resultsDeconvolution=resultsDeconvolution+[{'regressorName':regressorName,'time':timeIndicator,'betaWeight':betaWeightsAsArray[fileIndex]}]
		
			except ValueError:
			
				splitName=fileName.split('.')
				regressorName=splitName[0]
				resultsNonDeconvolution=resultsNonDeconvolution+[{'regressorName':regressorName,'data':betaWeightsAsArray[fileIndex]}]
	
	deconvolutionRegressorNames=list(set([element['regressorName'] for element in resultsDeconvolution]).union())
	
	reassembledResultsDeconvolution=[]
	for oneName in deconvolutionRegressorNames:
		theseEntries=[element for element in resultsDeconvolution if element['regressorName']==oneName]
		theseEntries=[[element['time'],element['betaWeight']] for element in theseEntries]
		theseEntries.sort()
		reassembledResultsDeconvolution=reassembledResultsDeconvolution+[{'regressorName':oneName,'data':theseEntries}]
		
	if showPlots:
		
		colors=['b','g','r','c','m','y','k','.75','.5','.25']
		
		fig = pl.figure(figsize = (25*figureScaler,15*figureScaler))
		s = fig.add_subplot(1,1,1)
		s.set_title(obsPlusSessionString+', '+analysisName)
	
		for oneIndex,oneRegressor in enumerate(reassembledResultsDeconvolution):
			pl.plot([element[0] for element in oneRegressor['data']], [element[1] for element in oneRegressor['data']],linewidth=2.,color=colors[oneIndex], label=oneRegressor['regressorName'])
	
		pl.legend()
		pl.show()
		
	commandline.ShellCmd('mkdir '+outputFolder)
	commandline.ShellCmd('mkdir '+outputFolder+'/'+obsPlusSessionString+'_'+analysisName)
	for oneRegressor in reassembledResultsDeconvolution:
		numpy.savetxt(outputFolder+'/'+obsPlusSessionString+'_'+analysisName+'/'+oneRegressor['regressorName']+'.txt',oneRegressor['data'],fmt='%20.10f\t%20.10f')
		
	for oneRegressor in resultsNonDeconvolution:
		numpy.savetxt(outputFolder+'/'+obsPlusSessionString+'_'+analysisName+'/'+oneRegressor['regressorName']+'.txt',oneRegressor['data'],fmt='%20.10f')

def identifySwitches(timeCoursePath,eventsPath,figuresPath,behavFilePath,miscStuffFilePath,categoryLimits,timeChunkPerPlot_ms):
	
	allFilenamesReportedSwitches=[element for element in os.listdir(eventsPath) if 'dirChangeReportEvent' in element]
	allFilenamesPhysicalSwitches=[element for element in os.listdir(eventsPath) if 'dirChangePhysEvent' in element]
	
	allFilenamesProjLenDiff=[element for element in os.listdir(timeCoursePath) if 'projectionLen' in element]
	allObs=[re.findall('observer([a-zA-Z0-9]*)session',element)[0] for element in allFilenamesProjLenDiff]
	uniqueObservers=list(set(allObs).union())
	
	behavFiles=[element for element in os.listdir(behavFilePath) if 'rivalryRG' in element]
	
	obsNotRunYet=[]
	allFilenamesPhysIdentitiesNoReturns=[element for element in os.listdir(eventsPath) if '_dirChangePhysIdentitiesNoReturns' in element]
	for oneObs in allObs:
		anyPhysIdentitiesNoReturnsFilesThisObs=[element for element in allFilenamesPhysIdentitiesNoReturns if 'observer_'+oneObs in element]
		if len(anyPhysIdentitiesNoReturnsFilesThisObs)==0:
			obsNotRunYet=obsNotRunYet+[oneObs]
	
	if len(obsNotRunYet)==0:
		userinteraction.printFeedbackMessage('Not identifying switches because it has already been done for all observers.')
	else:
		userinteraction.printFeedbackMessage('Identifying switches because there are new observers. These are as follows:')
		allInferredRates=[]
	
		for oneObs in uniqueObservers:
			allSessions=[re.findall('observer'+oneObs+'session([0-9]*)',element) for element in allFilenamesProjLenDiff]
			uniqueSessions=list(set([element[0] for element in allSessions if element]).union())
		
			for oneSession in uniqueSessions:
			
				obsPlusSessionString='observer_'+oneObs+'session'+oneSession
			
				reportedTimesFlattened=[]
				reportedIdentitiesFlattened=[]
				reportedTransitionDurationsFlattened=[]
				physicalTimesFlattened=[]
				physicalIdentitiesFlattened=[]
				physicalTransitionDurationsFlattened=[]
			
				transitionTimesAllTrials=[]
				transitionIdentitiesAllTrials=[]
				totTime=0
			
				reportedSwitchFile=[element for element in allFilenamesReportedSwitches if 'observer'+oneObs+'session'+oneSession+'_' in element]
				if reportedSwitchFile:
					reportedSwitches=filemanipulation.readDelimitedIntoArray(eventsPath+reportedSwitchFile[0],'\t')
				else:
					reportedSwitches=[]
				
				physicalSwitchFile=[element for element in allFilenamesPhysicalSwitches if 'observer'+oneObs+'session'+oneSession+'_' in element]
				if physicalSwitchFile:
					physicalSwitches=filemanipulation.readDelimitedIntoArray(eventsPath+physicalSwitchFile[0],'\t')
				else:
					physicalSwitches=[]
				
				originalBehavFile=[candidateFile for candidateFile in behavFiles if 'rivalryRG_observer_'+oneObs+'_session_'+oneSession in candidateFile][0]
				sessionKind=re.findall('.*_rivalryReplay_([0-9]*)_reportSwitchesProbes_([0-9]*)_time.*',originalBehavFile)[0]
			
				allFilenamesProjLenDiffThisObsSession=[element for element in allFilenamesProjLenDiff if 'observer'+oneObs+'session'+oneSession+'_' in element]

				for trialFile in allFilenamesProjLenDiffThisObsSession:
				
					trialNum=int(re.findall('trial_([0-9]*)_',trialFile)[0])
				
					projLenDifData=filemanipulation.readDelimitedIntoArray(timeCoursePath+trialFile,'\t')
					totTime=totTime+(projLenDifData[-1][0]-projLenDifData[0][0])
			
					indicesA=[index for index,element in enumerate(projLenDifData) if element[1]<=categoryLimits[0]]
					indicesB=[index for index,element in enumerate(projLenDifData) if element[1]>=categoryLimits[1]]
			
					projLenDifDataPlusPercept=[]
					for index,element in enumerate(projLenDifData):			#1: assign percept to time samples where pursuit is extreme enough
						if index in indicesA:
							projLenDifDataPlusPercept=projLenDifDataPlusPercept+[element+[-1]]
						elif index in indicesB:
							projLenDifDataPlusPercept=projLenDifDataPlusPercept+[element+[1]]
						else:
							projLenDifDataPlusPercept=projLenDifDataPlusPercept+[element+[0]]
			
					transitionTimes=[]
					for primaryIndex,triplet in enumerate(projLenDifDataPlusPercept):		#2: assign transition moments if opposite percepts immediately abutting.
						for limitIndex,minusOrNot in enumerate([-1,1]):
					
							if triplet[-1]==minusOrNot:
						
								for walkBackAndForth in [-1,1]:
									secondaryIndex=primaryIndex+walkBackAndForth
									#keepGoing=True
									if secondaryIndex>=0 and secondaryIndex<len(projLenDifDataPlusPercept):
										if projLenDifDataPlusPercept[secondaryIndex][-1]==-minusOrNot:
											candidateTransitionTime=float(triplet[0]+projLenDifDataPlusPercept[secondaryIndex][0])/2.
											if not candidateTransitionTime in transitionTimes:
												transitionTimes=transitionTimes+[candidateTransitionTime] #if our primaryIndex is right at a sharp boundary to the other percept: we mark the transition
											keepGoing=False
			
					for primaryIndex,triplet in enumerate(projLenDifDataPlusPercept):	#3: glue together periods of same-percept assignment if separated by only zeros, and assign transition moments to unassigned periods if flanked by opposite percepts
						if triplet[-1]==0:
							projLenDifDataPlusPercept[primaryIndex][-1]=2	#2 is a placeholder for: 0 but I've already checked this spot
							flankingVals=[]
							flankingIndices=[]
							for walkBackAndForth in [-1,1]:
								secondaryIndex=primaryIndex+walkBackAndForth
								keepGoing=True
								while secondaryIndex>=0 and secondaryIndex<len(projLenDifDataPlusPercept) and keepGoing:
									assignedPerc=projLenDifDataPlusPercept[secondaryIndex][-1]
							
									if assignedPerc==0:
										projLenDifDataPlusPercept[secondaryIndex][-1]=2
									else:
										flankingVals=flankingVals+[assignedPerc]
										flankingIndices=flankingIndices+[secondaryIndex]
										keepGoing=False
									secondaryIndex=secondaryIndex+walkBackAndForth
					
							if flankingVals==[-1,1] or flankingVals==[1,-1]:
						
								bestTime=float(projLenDifDataPlusPercept[flankingIndices[0]][0]+projLenDifDataPlusPercept[flankingIndices[1]][0])/2.#it seems to mark a switch halfway the transition period, if there is a transition period.
						
								# startTime=projLenDifDataPlusPercept[flankingIndices[0]+1][0]
								# 
								# fitParams=numpy.polyfit([projLenDifDataPlusPercept[elementin][0]-startTime for elementin in range(flankingIndices[0]+1,flankingIndices[1])],[projLenDifDataPlusPercept[elementin][1] for elementin in range(flankingIndices[0]+1,flankingIndices[1])],1)
								# 
								# offset=fitParams[1]
								# slope=fitParams[0]
								# 
								# absVal=numpy.inf
								# bestTime=startTime
								# for timeVal in numpy.linspace(0,(projLenDifDataPlusPercept[flankingIndices[-1]-1][0]-startTime),500):
								# 	absValNew=numpy.abs(offset+slope*timeVal)
								# 	if absValNew<absVal:
								# 		absVal=absValNew
								# 		bestTime=timeVal+startTime
						
								transitionTimes=transitionTimes+[bestTime]
						
							elif flankingVals==[-1,-1] or flankingVals==[-1]:
								for paintOverIndex in flankingIndices:
									projLenDifDataPlusPercept[paintOverIndex][-1]=-1
							elif flankingVals==[1,1] or flankingVals==[1]:
								for paintOverIndex in flankingIndices:
									projLenDifDataPlusPercept[paintOverIndex][-1]=1
				
					transitionTimes.sort()
				
					transitionIdentities=[]
					for transitionTime in transitionTimes:
						perSamplePerceptsAfter=[element[-1] for element in projLenDifDataPlusPercept if element[0]>transitionTime and element[-1] in [-1,1]]
						transitionIdentities=transitionIdentities+[perSamplePerceptsAfter[0]]
				
					minDuration_ms=500.
					removeIndices=[]
					for thisIndex in range(len(transitionTimes)-1):
						if (transitionTimes[thisIndex+1]-transitionTimes[thisIndex])<minDuration_ms:
							removeIndices=removeIndices+[thisIndex,thisIndex+1]
				
					transitionTimes=[element for thisIndex,element in enumerate(transitionTimes) if not thisIndex in removeIndices]
					transitionIdentities=[element for thisIndex,element in enumerate(transitionIdentities) if not thisIndex in removeIndices]
			
					transitionTimesAllTrials=transitionTimesAllTrials+[transitionTimes]
					transitionIdentitiesAllTrials=transitionIdentitiesAllTrials+[transitionIdentities]
				
					#UPDATED APRIL 11 2019 TO DO TWO THINGS: MARK THE TIME HALFWAY THE TRANSITION PERIOD (IF THERE IS ONE) INSTEAD OF AT THE END, AND STORE A SEPARATE ARRAY OF TRANSITION DURATIONS (in deleteReturnsMixDurations)
					theseReportedSwitchTimes=[[element[0]*1000.,element[-1]] for element in reportedSwitches if element[0]*1000.>=projLenDifData[0][0] and element[0]*1000.<=projLenDifData[-1][0]]
					if theseReportedSwitchTimes:
						theseReportedSwitchTimes=theseReportedSwitchTimes[1:]	#people also report the first percept, which is not a switch
						deleteMixAndReturns=[]
						deleteMixAndReturnsPerceptIdentities=[]
						deleteReturnsMixDurations=[]
						latestExclusive=666
						currentlyInMixture=False
						latestMixStartTime=666
						
						for oneCandidate in theseReportedSwitchTimes:
							thisPerc=oneCandidate[-1]
							if thisPerc in [0.,1.]:
								newExclusive=thisPerc
								if not latestExclusive==newExclusive:
									
									if currentlyInMixture:
										deleteMixAndReturns=deleteMixAndReturns+[(oneCandidate[0]+latestMixStartTime)/2.]
										deleteReturnsMixDurations=deleteReturnsMixDurations+[oneCandidate[0]-latestMixStartTime]
									else:
										deleteMixAndReturns=deleteMixAndReturns+[oneCandidate[0]]
										deleteReturnsMixDurations=deleteReturnsMixDurations+[0]
										
									deleteMixAndReturnsPerceptIdentities=deleteMixAndReturnsPerceptIdentities+[-(oneCandidate[-1]*2-1)]	#*2-1 stuff to get it on a -1 vs 1 scale
									latestExclusive=newExclusive
									currentlyInMixture=False
									
							elif thisPerc == 2.:
								if not currentlyInMixture:
									currentlyInMixture=True
									latestMixStartTime=oneCandidate[0]
						
						theseReportedSwitchTimes=deleteMixAndReturns
						theseReportedSwitchIdentities=deleteMixAndReturnsPerceptIdentities
						theseReportedTransitionDurations=deleteReturnsMixDurations
					else:
						theseReportedSwitchIdentities=[]
						theseReportedTransitionDurations=[]
							
					thesePhysicalSwitchTimes=[[element[0]*1000.,element[-1]] for element in physicalSwitches if element[0]*1000.>=projLenDifData[0][0] and element[0]*1000.<=projLenDifData[-1][0]]
					if thesePhysicalSwitchTimes:
						deleteMixAndReturns=[]
						deleteMixAndReturnsPerceptIdentities=[]
						deleteReturnsMixDurations=[]
						latestExclusive=666
						currentlyInMixture=False
						latestMixStartTime=666
						
						for oneCandidate in thesePhysicalSwitchTimes:
							thisPerc=oneCandidate[-1]
							if thisPerc in [0.,1.]:
								newExclusive=thisPerc
								if not latestExclusive==newExclusive:
									
									if currentlyInMixture:
										deleteMixAndReturns=deleteMixAndReturns+[(oneCandidate[0]+latestMixStartTime)/2.]
										deleteReturnsMixDurations=deleteReturnsMixDurations+[oneCandidate[0]-latestMixStartTime]
										
									else:
										
										deleteMixAndReturns=deleteMixAndReturns+[oneCandidate[0]]
										deleteReturnsMixDurations=deleteReturnsMixDurations+[0]
										
									deleteMixAndReturnsPerceptIdentities=deleteMixAndReturnsPerceptIdentities+[oneCandidate[-1]*2-1]	#*2-1 stuff to get it on a -1 vs 1 scale
									latestExclusive=newExclusive
									currentlyInMixture=False
									
							elif thisPerc==2:
								
								if not currentlyInMixture:
									currentlyInMixture=True
									latestMixStartTime=oneCandidate[0]
								
						thesePhysicalSwitchTimes=deleteMixAndReturns
						thesePhysicalSwitchIdentities=deleteMixAndReturnsPerceptIdentities
						thesePhysicalTransitionDurations=deleteReturnsMixDurations
					else:
						thesePhysicalSwitchIdentities=[]
						thesePhysicalTransitionDurations=[]
					
					reportedTimesFlattened=reportedTimesFlattened+theseReportedSwitchTimes
					reportedIdentitiesFlattened=reportedIdentitiesFlattened+theseReportedSwitchIdentities
					reportedTransitionDurationsFlattened=reportedTransitionDurationsFlattened+theseReportedTransitionDurations
					
					physicalTimesFlattened=physicalTimesFlattened+thesePhysicalSwitchTimes
					physicalIdentitiesFlattened=physicalIdentitiesFlattened+thesePhysicalSwitchIdentities
					physicalTransitionDurationsFlattened=physicalTransitionDurationsFlattened+thesePhysicalTransitionDurations
				
					firstTimePoint=projLenDifData[0][0]
					lastTimePoint=projLenDifData[-1][0]

					startTimePoint=firstTimePoint
					while startTimePoint<lastTimePoint:

						endTimePoint=min([startTimePoint+timeChunkPerPlot_ms,lastTimePoint])
	
						fig = pl.figure(figsize = (12,12))
						s = fig.add_subplot(1,1,1)
						s.set_title(str(oneObs)+'_session_'+str(oneSession)+'_trial_'+str(trialNum))

						projLenDifDataThisPlot=[element for element in projLenDifData if (element[0]>=startTimePoint and element[0]<=endTimePoint)]
						inferredPerceptDataThisPlot=[[element[0],element[-1]] for element in projLenDifDataPlusPercept if (element[0]>=startTimePoint and element[0]<=endTimePoint)]
						inferredPerceptDataThisPlot2to0=[]
						for element in inferredPerceptDataThisPlot:
							if element[1]==2:
								inferredPerceptDataThisPlot2to0=inferredPerceptDataThisPlot2to0+[[element[0],0]]
							else:
								inferredPerceptDataThisPlot2to0=inferredPerceptDataThisPlot2to0+[element]
						inferredPerceptDataThisPlot=inferredPerceptDataThisPlot2to0	
				
						if not(projLenDifDataThisPlot):
					
							userinteraction.printFeedbackMessage("not making plot for "+str(oneObs)+'_session_'+str(oneSession)+'trial'+str(trialNum)+" interval "+str(startTimePoint)+" to "+str(endTimePoint)+" because it contains no data")
					
						else:
					
							#pl.plot([element[0] for element in projLenDifDataThisPlot],[element[1] for element in projLenDifDataThisPlot], color = 'k', linewidth=1.2)
							pl.plot([element[0] for element in projLenDifDataThisPlot],[element[1] for element in projLenDifDataThisPlot], color = 'k', linewidth=1.2,label='proj len diff')
							pl.plot([element[0] for element in inferredPerceptDataThisPlot],[element[1] for element in inferredPerceptDataThisPlot], color = [.5,.5,.5], linewidth=1.2,label='inferred perc')
					
							inferredIndicesThisChunk=[index for index,element in enumerate(transitionTimes) if (element>=startTimePoint and element<=endTimePoint)]
							reportedIndicesThisChunk=[index for index,element in enumerate(theseReportedSwitchTimes) if (element>=startTimePoint and element<=endTimePoint)]
							physicalIndicesThisChunk=[index for index,element in enumerate(thesePhysicalSwitchTimes) if (element>=startTimePoint and element<=endTimePoint)]
						
							inferredTimesThisChunk=[transitionTimes[index] for index in inferredIndicesThisChunk]
							reportedTimesThisChunk=[theseReportedSwitchTimes[index] for index in reportedIndicesThisChunk]
							physicalTimesThisChunk=[thesePhysicalSwitchTimes[index] for index in physicalIndicesThisChunk]
						
							inferredIdentitiesThisChunk=[transitionIdentities[index] for index in inferredIndicesThisChunk]
							reportedIdentitiesThisChunk=[theseReportedSwitchIdentities[index] for index in reportedIndicesThisChunk]
							physicalIdentitiesThisChunk=[thesePhysicalSwitchIdentities[index] for index in physicalIndicesThisChunk]
					
							colors=['r','g','b']
							yRanges=[[-1.,-.3333],[-.3333,.3333],[.3333,1.]]
							labels=['inferred','reported','physical']
							identityList=[inferredIdentitiesThisChunk,reportedIdentitiesThisChunk,physicalIdentitiesThisChunk]
						
							for plotIndex,stuff in enumerate([inferredTimesThisChunk,reportedTimesThisChunk,physicalTimesThisChunk]):
								[pl.plot([element,element],[yRanges[plotIndex][0],yRanges[plotIndex][1]], color = colors[plotIndex], linewidth=2.,label=labels[plotIndex]) if elementIndex==0 else pl.plot([element,element],[yRanges[plotIndex][0],yRanges[plotIndex][1]], color = colors[plotIndex], linewidth=2.) for elementIndex,element in enumerate(stuff)]
								[pl.scatter([element],[yRanges[plotIndex][int((identityList[plotIndex][elementIndex]+1)/2)]], color = colors[plotIndex]) for elementIndex,element in enumerate(stuff)]
							
							pl.legend()
					
						s.set_xlim(xmin =startTimePoint, xmax = endTimePoint)
						s.set_ylim(ymin =-1.05, ymax = 1.05)
				
						pl.savefig(figuresPath+str(oneObs)+'_session_'+str(oneSession)+'_trial_'+str(trialNum)+"_"+str(startTimePoint)+"-"+str(endTimePoint)+'_switches_real_and_inferred.pdf')
						pl.close()
				
						startTimePoint=endTimePoint
					
				inferredTimesFlattened=[item for sublist in transitionTimesAllTrials for item in sublist]
				inferredIdentitiesFlattened=[item for sublist in transitionIdentitiesAllTrials for item in sublist]
				# reportedTimesFlattened=[element[0]*1000 for element in reportedSwitches]
				# physicalTimesFlattened=[element[0]*1000 for element in physicalSwitches]
	
				inferredPerSecond=1000*len(inferredTimesFlattened)/totTime
				reportedPerSecond=1000*len(reportedTimesFlattened)/totTime
				physicalPerSecond=1000*len(physicalTimesFlattened)/totTime
			
				numBins=25
				binBoundaries=numpy.linspace(-1000./inferredPerSecond,1000./inferredPerSecond,numBins+1)
				allSwitchKinds=[inferredTimesFlattened,reportedTimesFlattened,physicalTimesFlattened]
				eventStrings=['inferred','reported','physical']
			
				fig = pl.figure(figsize = (12,20))
				for referenceIndex,referenceSwitches in enumerate(allSwitchKinds):
				
					s = fig.add_subplot(3,1,referenceIndex+1)
					s.set_title('Ref event: '+eventStrings[referenceIndex])
				
					numRefSwitches=float(len(referenceSwitches))
					for comparisonIndex,comparisonSwitches in enumerate(allSwitchKinds):
						xVals=[]
						yVals=[]
						for oneBinIndex in range(numBins):
							theseBoundaries=[binBoundaries[oneBinIndex],binBoundaries[oneBinIndex+1]]
							xVal=numpy.average(theseBoundaries)
							yVal=0.
							for oneRefTime in referenceSwitches:
								numInBin=len([element for element in comparisonSwitches if element>=(oneRefTime+theseBoundaries[0]) and element<(oneRefTime+theseBoundaries[1])])
							
								yVal=yVal+float(numInBin)/numRefSwitches
						
							if comparisonIndex==referenceIndex and 0>=theseBoundaries[0] and 0<theseBoundaries[1]:
								yVal=yVal-1.
							
							xVals=xVals+[xVal]
							yVals=yVals+[yVal]
					
						pl.plot(xVals,yVals, color = colors[comparisonIndex], linewidth=1.,label=eventStrings[comparisonIndex])

					pl.legend()

					s.set_xlim(xmin =binBoundaries[0], xmax = binBoundaries[-1])
					s.set_ylim(ymin =0., ymax = .8)

				pl.savefig(figuresPath+str(oneObs)+'_session_'+str(oneSession)+'_switch_related_tallies.pdf')
				pl.close()
			
				allInferredRates=allInferredRates+[[oneObs,sessionKind,inferredPerSecond,reportedPerSecond,physicalPerSecond]]
			
				numpy.save(miscStuffFilePath+obsPlusSessionString+'_switch_rate_comparison_data',[oneObs,sessionKind,inferredPerSecond,reportedPerSecond,physicalPerSecond])
				
				forOutput=[physicalTimesFlattened,reportedTimesFlattened,inferredTimesFlattened]
				forOutputIdentities=[physicalIdentitiesFlattened,reportedIdentitiesFlattened,inferredIdentitiesFlattened]
				namesForOutput=['dirChangePhysTimesNoReturns','dirChangeReportTimesNoReturns','dirChangeInferredTimesNoReturns']
				namesForOutputIdentities=['dirChangePhysIdentitiesNoReturns','dirChangeReportIdentitiesNoReturns','dirChangeInferredIdentitiesNoReturns']
			
				for aapje in range(3):
					numpy.savetxt(eventsPath+obsPlusSessionString+'_'+namesForOutput[aapje]+'.txt',[element/1000. for element in forOutput[aapje]],fmt='%20.10f')
					numpy.savetxt(eventsPath+obsPlusSessionString+'_'+namesForOutputIdentities[aapje]+'.txt',[element for element in forOutputIdentities[aapje]],fmt='%20.10f')
				
				forOutputTransDurations=[physicalTransitionDurationsFlattened,reportedTransitionDurationsFlattened]
				namesForOutputTransDurations=['transDursPhysNoReturns','transDursReportNoReturns']
				for aapje in range(2):
					numpy.savetxt(eventsPath+obsPlusSessionString+'_'+namesForOutputTransDurations[aapje]+'.txt',[element for element in forOutputTransDurations[aapje]],fmt='%20.10f')
				
		#x,y,title
		interestingCombis=[[[('0','0'),3],[('0','0'),2],"Active rivalry inferred vs reported"],[[('0','0'),2],[('0','1'),2],"Passive rivalry inferred vs active rivalry inferred"],[[('1','0'),3],[('1','0'),4],"Active on-screen physical vs reported"],[[('1','0'),3],[('1','0'),2],"Active on-screen inferred vs reported"],[[('1','0'),4],[('1','0'),2],"Active on-screen inferred vs physical"],[[('1','0'),2],[('1','1'),2],"Passive on-screen inferred vs active on-screen inferred"]]
	
		fig = pl.figure(figsize = (20,20))
		for imageIndex,interestingCombi in enumerate(interestingCombis):
		
			s = fig.add_subplot(3,3,imageIndex+1)
			s.set_title(interestingCombi[-1])
		
			xData=[element[interestingCombi[0][1]] for element in allInferredRates if element[1]==interestingCombi[0][0]]
			yData=[element[interestingCombi[1][1]] for element in allInferredRates if element[1]==interestingCombi[1][0]]
		
			pl.scatter(xData,yData)
			pl.plot([0,1.1*max(xData+yData)],[0,1.1*max(xData+yData)])

			s.set_xlim(xmin =0, xmax = 1.1*max(xData+yData))
			s.set_ylim(ymin =0., ymax = 1.1*max(xData+yData))

		pl.savefig(figuresPath+'switch_rate_correlations.pdf')
		pl.close()
	
def plotReportedSwitchInfo(regressorPath,figurePath,behavioralFilePath,obsLeftOutOfS2S=[]):
	
	binNumber=40
	allFilenamesBehavioral=[element for element in os.listdir(behavioralFilePath) if 'rivalryRG' in element]
	allObs=[re.findall('observer_([a-zA-Z0-9]*)_session',element)[0] for element in allFilenamesBehavioral]
	uniqueObservers=list(set(allObs).union())
	
	allMixProps=[]
	allAvgTransDurs=[]
	allPropTransGT1000ms=[]
	allAvgS2SDurs=[]
	
	figAll = pl.figure(figsize = (25,25))
	for obsIndex,obs in enumerate(uniqueObservers):
		
		reportedRivalryFileThisObs=[element for element in os.listdir(behavioralFilePath) if 'observer_'+obs+'_session' in element and '_rivalryReplay_0_reportSwitchesProbes_0' in element][0]
		relevantSes=re.findall('session_([a-zA-Z0-9]*)_rivalryReplay',reportedRivalryFileThisObs)[0]
		
		switchFile='observer'+obs+'session'+relevantSes+'_dirChangeReportEvent.txt'
		trialStartFile='observer'+obs+'session'+relevantSes+'_trialOnset.txt'
		trialEndFile='observer'+obs+'session'+relevantSes+'_trialOffset.txt'
		
		[switchData,trialStartData,trialEndData]=[filemanipulation.readDelimitedIntoArray(regressorPath+fileName,'\t') for fileName in [switchFile,trialStartFile,trialEndFile]]
		
		trialIntervals=[[trialStartData[index],trialEndData[index]] for index in range(len(trialStartData))]
		
		startToStartDurs=[]
		startToEndDurs=[]
		transDurs=[]
		totDomDur=0
		totTransDur=0
		
		for trialInterval in trialIntervals:
			relevantSwitchData=[element for element in switchData if element[0]>trialInterval[0][0] and element[0]<trialInterval[1][0]]
			
			sample=relevantSwitchData[0]
			latestKey=sample[-1]
			latestKeyTime=sample[0]
			currentlyInMix=(latestKey==2.)
			if currentlyInMix:
				latestMixStartTime=latestKeyTime
				latestExclStartTimeStS=-10
				latestExclStartTimeStE=-10
				latestExclPerc=-10
			else:
				latestMixStartTime=-10
				latestExclStartTimeStS=latestKeyTime
				latestExclStartTimeStE=latestKeyTime
				latestExclPerc=latestKey
			
			for sample in relevantSwitchData[1:]:
				newKey=sample[-1]
				newKeyTime=sample[0]
				
				if newKey==2: #a mix key is pressed
				
					if not currentlyInMix:
						currentlyInMix=True
						startToEndDurs=startToEndDurs+[newKeyTime-latestExclStartTimeStE]
						latestMixStartTime=newKeyTime
						totDomDur=totDomDur+newKeyTime-latestExclStartTimeStE
					
				else: #a non-mix key is pressed
					
					if currentlyInMix:
						currentlyInMix=False
						transDurs=transDurs+[newKeyTime-latestMixStartTime]	#regardless of whether it's a return transition or not
						totTransDur=totTransDur+newKeyTime-latestMixStartTime
						
						latestExclStartTimeStE=newKeyTime
						
						if not(newKey==latestExclPerc):
							if not latestExclStartTimeStS==-10:
								startToStartDurs=startToStartDurs+[newKeyTime-latestExclStartTimeStS]
							latestExclStartTimeStS=newKeyTime
							latestExclPerc=newKey
					
					else:
						if not(newKey==latestExclPerc):
							if not latestExclStartTimeStS==-10:
								transDurs=transDurs+[0]
								startToStartDurs=startToStartDurs+[newKeyTime-latestExclStartTimeStS]
								startToEndDurs=startToEndDurs+[newKeyTime-latestExclStartTimeStE]
								totDomDur=totDomDur+newKeyTime-latestExclStartTimeStE
								latestExclStartTimeStE=newKeyTime
								latestExclStartTimeStS=newKeyTime
					
			if currentlyInMix:
				if not latestMixStartTime==-10:
					totTransDur=totTransDur+trialInterval[1][0]-latestMixStartTime
			else:
				if not latestExclStartTimeStE==-10:
					totDomDur=totDomDur+trialInterval[1][0]-latestExclStartTimeStE

		fig = pl.figure(figsize = (10,10))

		s = fig.add_subplot(1,1,1)
		
		if (totDomDur+totTransDur)>0:
			propTimeInMixture=totTransDur/(totDomDur+totTransDur)
			maxX=max(max(transDurs),max(startToStartDurs),max(startToEndDurs))
			plotTitle=obs+'; prop mix='+str(round(propTimeInMixture*1000)/1000)+'; avg mix dur='+str(round(numpy.average(transDurs)*1000)/1000)+'\navg s2s dur='+str(round(numpy.average(startToStartDurs)*1000)/1000)+'; avg s2e dur='+str(round(numpy.average(startToEndDurs)*1000)/1000)+'\nRGB=s2s,mix;s2e'
			binBoundaries=numpy.linspace(-.000001,maxX,binNumber)
		
			talliesStartToStart=[len([observation for observation in startToStartDurs if observation>binBoundaries[boundaryIndex] and observation<=binBoundaries[boundaryIndex+1]]) for boundaryIndex in range(binNumber-1)]
			talliesMix=[len([observation for observation in transDurs if observation>binBoundaries[boundaryIndex] and observation<=binBoundaries[boundaryIndex+1]]) for boundaryIndex in range(binNumber-1)]
			talliesStartToEnd=[len([observation for observation in startToEndDurs if observation>binBoundaries[boundaryIndex] and observation<=binBoundaries[boundaryIndex+1]]) for boundaryIndex in range(binNumber-1)]
		
			xData=[numpy.average([binBoundaries[boundaryIndex],binBoundaries[boundaryIndex+1]]) for boundaryIndex in range(binNumber-1)]
			
		else:
			
			userinteraction.printFeedbackMessage("WATCH OUT: THERE ARE NO START TO START DURS AT ALL FOR OBSERVER "+obs+".")
			propTimeInMixture=0.
			maxX=1
			plotTitle="obs+'; no data at all. This is rubbish"
			
			xData=[0,0]
			talliesStartToStart=[0,0]
			talliesMix=[0,0]
			talliesStartToEnd=[0,0]
		
		allMixProps=allMixProps+[[propTimeInMixture,obs]]
		allAvgTransDurs=allAvgTransDurs+[[numpy.average(transDurs),obs]]
		allPropTransGT1000ms=allPropTransGT1000ms+[[float(len([element for element in transDurs if element>1.]))/float(len(transDurs)),obs]]
		if not obs in obsLeftOutOfS2S:
			allAvgS2SDurs=allAvgS2SDurs+[[numpy.average(startToStartDurs),obs]]
		
		s.set_title(plotTitle)
		
		pl.plot(xData,talliesStartToStart,'r')
		pl.plot(xData,talliesMix,'g')
		pl.plot(xData,talliesStartToEnd,'b')

		s.set_xlim(xmin =0, xmax = maxX)
		s.set_ylim(ymin =0., ymax = max([max(talliesMix),max(talliesStartToStart),max(talliesStartToEnd)])+10)
		
		s.set_xlabel('Time (s)')
		s.set_ylabel('Tally')

		pl.savefig(figurePath+obs+'_reported_rivalry_dynamics.pdf')
		pl.close()
		
		s = figAll.add_subplot(6,6,obsIndex+1)
		s.set_title(plotTitle)
		
		pl.plot(xData,talliesStartToStart,'r')
		pl.plot(xData,talliesMix,'g')
		pl.plot(xData,talliesStartToEnd,'b')

		s.set_xlim(xmin =0, xmax = maxX)
		s.set_ylim(ymin =0., ymax = max([max(talliesMix),max(talliesStartToStart),max(talliesStartToEnd)])+10)
		
		s.set_xlabel('Time (s)')
		s.set_ylabel('Tally')
	
	pl.tight_layout()
	pl.savefig(figurePath+'acrossObs_reported_rivalry_dynamics.pdf')
	pl.close()
	
	fig = pl.figure(figsize = (10,10))
	s = fig.add_subplot(1,1,1)
	s.set_title('All observers')
	
	allMixProps.sort()
	allMixProps=[[0,'']]+allMixProps
	xVals=[element[0] for element in allMixProps]
	yVals=[float(thisVal)/(len(allMixProps)-1) for thisVal in range(len(allMixProps))]
	
	for gridX in [.3,.4,.5]:
		pl.plot([gridX,gridX],[0,1],color='k')
		
	pl.plot(xVals,yVals)
	pl.scatter(xVals,yVals)
	[pl.text(xVals[index]+.02, yVals[index]-.02, allMixProps[index][1], fontsize=12) for index in range(len(allMixProps))]
	
	s.set_xlim(xmin =0, xmax = 1.)
	s.set_ylim(ymin =0., ymax = 1.)
	
	s.set_xlabel('Mixture proportion')
	s.set_ylabel('Prop obs mix prop <= this value')

	pl.savefig(figurePath+'acrossObs_mix_proportions.pdf')
	pl.close()
	
	fig = pl.figure(figsize = (10,10))
	s = fig.add_subplot(1,1,1)
	s.set_title('All observers')
	
	allAvgTransDurs.sort()
	allAvgTransDurs=[[0,'']]+allAvgTransDurs
	xVals=[element[0] for element in allAvgTransDurs]
	yVals=[float(thisVal)/(len(allAvgTransDurs)-1) for thisVal in range(len(allAvgTransDurs))]
	
	for gridX in [1,2,3]:
		pl.plot([gridX,gridX],[0,1],color='k')

	pl.plot(xVals,yVals)
	pl.scatter(xVals,yVals)
	[pl.text(xVals[index]+.02, yVals[index]-.02, allAvgTransDurs[index][1], fontsize=12) for index in range(len(allAvgTransDurs))]
	
	s.set_xlim(xmin =0, xmax = max(xVals)+.5)
	s.set_ylim(ymin =0., ymax = 1.)
	
	s.set_xlabel('Average mixture duration (s)')
	s.set_ylabel('Avg mix dur <= this value')

	pl.savefig(figurePath+'acrossObs_mix_durs.pdf')
	pl.close()
	
	fig = pl.figure(figsize = (10,10))
	s = fig.add_subplot(1,1,1)
	s.set_title('All observers')
	
	allPropTransGT1000ms.sort()
	allPropTransGT1000ms=[[0,'']]+allPropTransGT1000ms
	xVals=[element[0] for element in allPropTransGT1000ms]
	yVals=[float(thisVal)/(len(allPropTransGT1000ms)-1) for thisVal in range(len(allPropTransGT1000ms))]
	
	pl.plot([.5,.5],[0,1],color='k')
	
	pl.plot(xVals,yVals)
	pl.scatter(xVals,yVals)
	[pl.text(xVals[index]+.02, yVals[index]-.02, allPropTransGT1000ms[index][1], fontsize=12) for index in range(len(allPropTransGT1000ms))]
	
	s.set_xlim(xmin =0, xmax = 1.)
	s.set_ylim(ymin =0., ymax = 1.)
	
	s.set_xlabel('Prop transition durations > 1000 ms')
	s.set_ylabel('Prop observers with score <= this value')

	pl.savefig(figurePath+'acrossObs_long_mix_props.pdf')
	pl.close()
		
	fig = pl.figure(figsize = (10,10))
	s = fig.add_subplot(1,1,1)
	s.set_title('All observers')
	
	allAvgS2SDurs.sort()
	allAvgS2SDurs=[[0,'']]+allAvgS2SDurs
	xVals=[element[0] for element in allAvgS2SDurs]
	yVals=[float(thisVal)/(len(allAvgS2SDurs)-1) for thisVal in range(len(allAvgS2SDurs))]
	
	for gridX in [1,2,3,4,5]:
		pl.plot([gridX,gridX],[0,1],color='k')
		
	for gridY in [.33,.5,.66]:
		pl.plot([0,max(xVals)+.5],[gridY,gridY],color='k')
	
	pl.plot(xVals,yVals)
	pl.scatter(xVals,yVals)
	[pl.text(xVals[index]+.02, yVals[index]-.02, allAvgS2SDurs[index][1], fontsize=12) for index in range(len(allAvgS2SDurs))]
	
	s.set_xlim(xmin =0, xmax = max(xVals)+.5)
	s.set_ylim(ymin =0., ymax = 1.)
	
	s.set_xlabel('Avg s2s duration (s)')
	s.set_ylabel('Prop observers with score <= this value')

	pl.savefig(figurePath+'acrossObs_s2s_durs.pdf')
	pl.close()
	
	numpy.save(figurePath+'acrossObs_s2s_durs_thedata',allAvgS2SDurs)