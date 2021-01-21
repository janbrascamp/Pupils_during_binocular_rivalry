#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Jan Brascamp on 2017-06-29.
Copyright (c) 2017 __MyCompanyName__. All rights reserved.

THIS IS THE VARIANT THAT WAS CREATED AFTER CONSULTING WITH TOMAS AND GILLES IN SEPTEMBER 2020, TO MAKE IT SO THAT THE PROCESSING ORDER IS blinks -> filter -> z-score -> concatenate, AND TO MAKE IT SO THAT THE FOLLOWING APPROACH IS USED TO FILTERING:

"""

import sys
import os
import numpy
import re
import scipy as sp
import scipy.signal
import pickle
from scipy.stats import ttest_1samp

from IPython import embed as shell

import matplotlib
import matplotlib.pyplot as pl

import helper_funcs.commandline	as commandline
import helper_funcs.analysis as analysis
import helper_funcs.userinteraction as userinteraction
import helper_funcs.filemanipulation as filemanipulation

# import seaborn as sn
# sn.set(style="ticks")

import FIRDeconvolution
from scipy import optimize

def mySurface(x,y,offset,coeffX,coeffY,offsetXsquared,offsetYsquared,coeffXsquared,coeffYsquared):

	return offset+x*coeffX+y*coeffY+pow(offsetXsquared,2)*coeffXsquared+pow(offsetYsquared,2)*coeffYsquared

def squareErrorFuncMySurface(x,*args):

	offset,coeffX,coeffY,offsetXsquared,offsetYsquared,coeffXsquared,coeffYsquared=x
	[xVals,yVals,zVals]=args

	#allFitParams=[offset,coeffX,coeffY,coeffXY,coeffXsquared,coeffYsquared]
	
	sumSquareErrors=0

	for observationIndex in range(len(xVals)):

		predictedZval=mySurface(xVals[observationIndex],yVals[observationIndex],offset,coeffX,coeffY,offsetXsquared,offsetYsquared,coeffXsquared,coeffYsquared)

		sumSquareErrors=sumSquareErrors+pow(zVals[observationIndex]-predictedZval,2)

	return sumSquareErrors

trialStartCode=0
trialEndCode=1
dirChangeReportedCode=2
detectionDotCode=3
keyPressedCode=4
missedCode=5
dirChangedCode=6

trialDurS=60.
waitDurS=1. #following drift correction or calibration. THIS SEEMS OUTDATED
colorProps=[.5]

subtractBaselineForPlots=False

simulatedTimeGapForMerging_s=1200		#between the start of the final trial of the previous session and the start of the first trial of the next

minNumEvs=5 		#if a regressor has fewer than this number of events, then don't try.

#myPath='/Users/janbrascamp/Documents/Experiments/pupil_switch/fall_18_dontcorrectforshortening/data/'
myPath='/Users/janbrascamp/Dropbox/__FS20/Experiments/fall_18_dontcorrectforshortening/data/'
miscStuffSubFolder='other'
figuresSubFolder='figures'
behavioralSubFolder='behavioral'
eyeSubFolder='eye'
beforeMergingFolder='before_merging'

#------merge behavioral and eye files so that the rest of the processing stream doesn't notice that there's more than 1 file per condition
allFilenamesBehavioralBeforeMergingAllObservers=[element for element in os.listdir(myPath+behavioralSubFolder+'/'+beforeMergingFolder)  if '.npy' in element]
theObservers=[re.findall('observer_([a-zA-Z0-9]*)_session', element)[0] for element in allFilenamesBehavioralBeforeMergingAllObservers]
theObservers=list(set(theObservers).union())

for oneObserver in theObservers:

	allFilenamesBehavioralBeforeMerging=[element for element in os.listdir(myPath+behavioralSubFolder+'/'+beforeMergingFolder)  if 'observer_'+oneObserver+'_session' in element]
	allFilenamesBehavioralBeforeMerging.sort()
	identifyingTriplets=[re.findall('_([0-9]*)_rivalryReplay_([0-9]*)_reportSwitchesProbes_([0-9]*)_time_', element)[0] for element in allFilenamesBehavioralBeforeMerging]
	identifyingTriplets.sort()

	uniqueConditions=[]
	for candidate in identifyingTriplets:
		if not [candidate[1],candidate[2]] in uniqueConditions:
			uniqueConditions=uniqueConditions+[[candidate[1],candidate[2]]]
	
	seshNumbersPerCondition=[[element[0] for element in identifyingTriplets if [element[1],element[2]]==uniqueCondition] for uniqueCondition in uniqueConditions]
	
	#merge behavioral
	allMergedFiles=[element for element in os.listdir(myPath+behavioralSubFolder)  if '.npy' in element]
	fileNameComponents=re.findall('(.*session_)[0-9]*(_rivalryReplay_)[0-9]*(_reportSwitchesProbes_)[0-9]*(_.*)',allFilenamesBehavioralBeforeMerging[0])[0]
	for oneConditionIndex,oneSetOfNumbers in enumerate(seshNumbersPerCondition):
		
		outputFileName=fileNameComponents[0]+str(oneConditionIndex)+fileNameComponents[1]+uniqueConditions[oneConditionIndex][0]+fileNameComponents[2]+uniqueConditions[oneConditionIndex][1]+fileNameComponents[3]
		
		if not outputFileName in allMergedFiles:
			allData=numpy.array([])
			for oneNumberIndex,oneNumber in  enumerate(oneSetOfNumbers):
				theFileName=[element for element in allFilenamesBehavioralBeforeMerging if 'session_'+oneNumber+'_rivalryReplay' in element][0]
				newData=numpy.load(myPath+behavioralSubFolder+'/'+beforeMergingFolder+'/'+theFileName)
				
				newDataTrialStartPoss=[index for index in range(len(newData)) if newData[index][0]==trialStartCode]
				
				if oneNumberIndex>0:		#this is to shift subsequent time points before collating, so that it seems like there's simulatedTimeGapForMerging_s separating trial onsets
					oldDataLastTrialTime=newDataLastTrialTime
					newDataFirstTrialTime=newData[newDataTrialStartPoss[1]][1]	#not 0 because that's more like a drift correction start
					
					actualTimeGap_s=newDataFirstTrialTime-oldDataLastTrialTime
					timeCorrectionRequired=simulatedTimeGapForMerging_s-actualTimeGap_s
					newDataLastTrialTime=newData[newDataTrialStartPoss[-1]][1]
					newData=numpy.array([[element[0],element[1]+timeCorrectionRequired]+element[2:] for element in newData])

				else:
					
					newDataLastTrialTime=newData[newDataTrialStartPoss[-1]][1]
				
				allData=numpy.append(allData,newData)
			numpy.save(myPath+behavioralSubFolder+'/'+outputFileName,allData)
		else:
			print 'not merging behavioral file '+outputFileName+' because already done'
			
	#merge eye
	allFilenamesEyeBeforeMerging=[element for element in os.listdir(myPath+eyeSubFolder+'/'+beforeMergingFolder)  if oneObserver in element]
	allMergedAscFiles=[element for element in os.listdir(myPath+eyeSubFolder)  if '.asc' in element]
	
	for oneConditionIndex,oneSetOfNumbers in enumerate(seshNumbersPerCondition):
		
		exampleEDFfileName=[element for element in allFilenamesEyeBeforeMerging if oneObserver+'C'+oneSetOfNumbers[0]+'_' in element and not('.asc' in element)][0]

		thisSampleOutName=oneObserver+'C'+str(oneConditionIndex)+'_'+exampleEDFfileName.split('_')[1][0]+'_s.asc'
		thisEventOutName=oneObserver+'C'+str(oneConditionIndex)+'_'+exampleEDFfileName.split('_')[1][0]+'_e.asc'
		
		if not thisSampleOutName in allMergedAscFiles:
		
			allSampleData=[]
			allEventData=[]
		
			for counter,oneNumber in  enumerate(oneSetOfNumbers):
			
				thisEDFfileName=[element for element in allFilenamesEyeBeforeMerging if oneObserver+'C'+oneNumber+'_' in element and not('.asc' in element)][0]
			
				if '.edf' in thisEDFfileName:
					thisEDFfileWithoutEdf=thisEDFfileName[:-4]
				else:
					thisEDFfileWithoutEdf=thisEDFfileName 
			
				commandline.ShellCmd('edf2asc -s '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileName)
				commandline.ShellCmd('mv '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'.asc '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'_s.asc')
				commandline.ShellCmd('edf2asc -e '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileName)
				commandline.ShellCmd('mv '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'.asc '+myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'_e.asc')
			
				newSampleData=filemanipulation.readDelimitedIntoArray(myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'_s.asc','\t')
				#time, x, y, pupil size, x, y, pupil size
				
				newEventData=filemanipulation.readDelimitedIntoArray(myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'_e.asc','\t')
			
				# #------correct for foreshortening before merging, by subtracting out a fitted 2D surface------
				
				# newDataTrialStartTimes_ms=[int(re.findall('([0-9]*) *Trial ([0-9]*) started',newEventData[index][-1])[0][0]) for index in range(len(newEventData)) if re.findall('([0-9]*) *Trial ([0-9]*) started',newEventData[index][-1])]
				# newDataTrialStartIndices=[[thisIndex for thisIndex,thisSample in enumerate(newSampleData) if int(thisSample[0])==thisTrialStartTime][0] for thisTrialStartTime in newDataTrialStartTimes_ms]
				# 
				# interSampleInterval_ms=int(newSampleData[1][0])-int(newSampleData[0][0])
				# allWithinTrialIndices=[]
				# for newDataTrialStartIndex in newDataTrialStartIndices:
				# 	allWithinTrialIndices=allWithinTrialIndices+range(newDataTrialStartIndex,newDataTrialStartIndex+int(trialDurS*1000/interSampleInterval_ms))
				# 
				# withinTrialSampleData=[newSampleData[thisIndex] for thisIndex in allWithinTrialIndices]
				# 
				# allValidXYpupilDataLeft=[[int(oneSample[0]),float(oneSample[1]),float(oneSample[2]),float(oneSample[3])] for oneSample in withinTrialSampleData if not float(oneSample[3])==0]	#if missing data then pupil size is 0
				# allValidXYpupilDataRight=[[int(oneSample[0]),float(oneSample[4]),float(oneSample[5]),float(oneSample[6])] for oneSample in withinTrialSampleData if not float(oneSample[6])==0]	#if missing data then pupil size is 0
				# 
				# f = pl.figure(figsize = (35,35))
				# f_corrected = pl.figure(figsize = (35,35))
				# 
				# coefficientsPerEye=[]
				# for leftRightIndex in [0,1]:
				# 	
				# 	thisEyeData=[allValidXYpupilDataLeft,allValidXYpupilDataRight][leftRightIndex]
				# 	
				# 	xDataForFit=[element[1] for element in thisEyeData]
				# 	
				# 	# xDataForFitMean=numpy.average(xDataForFit)	#we're going to discard outliers so that our fit isn't massively constrained by them
				# 	# xDataForFitStd=numpy.std(xDataForFit)
				# 	# lowerLimit=xDataForFitMean-xDataForFitStd*2.
				# 	# upperLimit=xDataForFitMean+xDataForFitStd*2.
				# 	# xDataForFit=[element for element in xDataForFit if element>lowerLimit and element<upperLimit]
				# 	xDataForFit.sort()
				# 	
				# 	binBorderValuesX=[xDataForFit[int(element)] for element in numpy.linspace(0,len(xDataForFit)-1,15)]
				# 	
				# 	binnedXYZcombis=[]
				# 	for xBinIndex in range(len(binBorderValuesX)-1):
				# 		
				# 		theseDataXOK=[element for element in thisEyeData if element[1]>binBorderValuesX[xBinIndex] and element[1]<=binBorderValuesX[xBinIndex+1]]
				# 		yDataForFit=[element[2] for element in theseDataXOK]
				# 		
				# 		# yDataForFitMean=numpy.average(yDataForFit)	#we're going to discard outliers so that our fit isn't massively constrained by them
				# 		# yDataForFitStd=numpy.std(yDataForFit)
				# 		# lowerLimit=yDataForFitMean-yDataForFitStd*2.
				# 		# upperLimit=yDataForFitMean+yDataForFitStd*2.
				# 		# yDataForFit=[element for element in yDataForFit if element>lowerLimit and element<upperLimit]
				# 		yDataForFit.sort()
				# 		
				# 		binBorderValuesY=[yDataForFit[int(element)] for element in numpy.linspace(0,len(yDataForFit)-1,15)]
				# 		
				# 		for yBinIndex in range(len(binBorderValuesY)-1):
				# 			
				# 			theseDataXandYOK=[element for element in theseDataXOK if element[2]>binBorderValuesY[yBinIndex] and element[2]<=binBorderValuesY[yBinIndex+1]]
				# 			binnedXYZcombis=binnedXYZcombis+[[numpy.average([element[1] for element in theseDataXandYOK]),numpy.average([element[2] for element in theseDataXandYOK]),numpy.average([element[3] for element in theseDataXandYOK])]]
				# 	
				# 	bestSqErrSoFar=numpy.inf
				# 	
				# 	fitAttemptCount=1
				# 	for xCoeff in [-1,1]:
				# 		for yCoeff in [-1,1]:
				# 			for xSquaredOffset in [1000.]:
				# 				for ySquaredOffset in [600.]:
				# 					for xSquaredCoeff in [-.1,.1]:
				# 						for ySquaredCoeff in [-.1,.1]:
				# 							
				# 							print 'Attempting fit '+str(fitAttemptCount)+' out of 16.'
				# 						
				# 							x0=numpy.asarray((0.,xCoeff,yCoeff,xSquaredOffset,ySquaredOffset,xSquaredCoeff,ySquaredCoeff))	#starting values
				# 							newFitResult = optimize.minimize(squareErrorFuncMySurface,x0,args=tuple([[element[0] for element in binnedXYZcombis],[element[1] for element in binnedXYZcombis],[element[2] for element in binnedXYZcombis]]),method='TNC',options={'maxiter':1000})
				# 						
				# 							newSqErr=newFitResult.fun
				# 						
				# 							if newSqErr<bestSqErrSoFar:
				# 								bestSqErrSoFar=newSqErr
				# 								fitResult=newFitResult
				# 							
				# 							fitAttemptCount=fitAttemptCount+1
				# 	
				# 	coeffs=fitResult.x[:]
				# 	coefficientsPerEye=coefficientsPerEye+[coeffs]
				# 	
				# 	fig = pl.figure()
				# 	fig.suptitle('Brighter = larger pupil; inside = observed; outside = fitted')
				# 	minZ=min([element[2] for element in binnedXYZcombis])
				# 	maxZ=max([element[2] for element in binnedXYZcombis])
				# 	
				# 	pl.scatter([element[0] for element in binnedXYZcombis], [element[1] for element in binnedXYZcombis],c=[[max([0,min([1,(mySurface(element[0],element[1],coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6])-minZ)/(maxZ-minZ)])]),0,0] for element in binnedXYZcombis],s=160)
				# 	pl.scatter([element[0] for element in binnedXYZcombis], [element[1] for element in binnedXYZcombis],c=[[(element[2]-minZ)/(maxZ-minZ),0,0] for element in binnedXYZcombis],edgecolors='w',s=70)
				# 	
				# 	pl.xlabel('x position')
				# 	pl.ylabel('y position')
				# 	
				# 	pl.savefig(myPath+figuresSubFolder+'/forshortening_3Ddata_'+thisEDFfileWithoutEdf+'_'+['L','R'][leftRightIndex]+'.pdf')
				# 	
				# 	pl.close()
				# 	
				# 	for xyIndex in [0,1]:
				# 		
				# 		theseXYdata=[[element[xyIndex+1],element[3],element[2-xyIndex]] for element in thisEyeData]	#x or y, pupil, y or x
				# 		
				# 		theseXYdata.sort()
				# 		
				# 		theseCoeffs=[coeffs[0],coeffs[1+xyIndex],coeffs[3+xyIndex],coeffs[5+xyIndex]]
				# 		otherCoeffs=[coeffs[0],coeffs[2-xyIndex],coeffs[4-xyIndex],coeffs[6-xyIndex]]
				# 		
				# 		binBorders=[int(element) for element in numpy.linspace(0,len(theseXYdata),10)]
				# 		
				# 		averagesPerBinXaxis=[]
				# 		averagesPerBinYaxis=[]
				# 		predictedAveragesPerBinYaxis=[]
				# 		for binBorderIndex in range(len(binBorders)-1):
				# 			
				# 			thisXaxisAverage=numpy.average([theseXYdata[thisIndex][0] for thisIndex in range(binBorders[binBorderIndex],binBorders[binBorderIndex+1])])
				# 			averagesPerBinXaxis=averagesPerBinXaxis+[thisXaxisAverage]
				# 			averagesPerBinYaxis=averagesPerBinYaxis+[numpy.average([theseXYdata[thisIndex][1] for thisIndex in range(binBorders[binBorderIndex],binBorders[binBorderIndex+1])])]
				# 		
				# 			thisremainingAverage=numpy.average([theseXYdata[thisIndex][2] for thisIndex in range(binBorders[binBorderIndex],binBorders[binBorderIndex+1])])	#the x coordinate if we're plotting y, or the y coordinate if we're plotting x
				# 			
				# 			predictedAveragesPerBinYaxis=predictedAveragesPerBinYaxis+[mySurface(thisXaxisAverage,thisremainingAverage,theseCoeffs[0],theseCoeffs[1],otherCoeffs[1],theseCoeffs[2],otherCoeffs[2],theseCoeffs[3],otherCoeffs[3])]
				# 			
				# 		pl.figure(f.number)	
				# 		s=f.add_subplot(2,2,xyIndex+1+leftRightIndex*2)
				# 		
				# 		s.set_title(['Left eye', 'Right eye'][leftRightIndex])
				# 		
				# 		xAxisMean=numpy.average([element[0] for element in theseXYdata])
				# 		xAxisStd=numpy.std([element[0] for element in theseXYdata])
				# 		
				# 		pl.scatter([element[0] for element in theseXYdata if (element[0]>xAxisMean-(2.5*xAxisStd) and element[0]<xAxisMean+(2.5*xAxisStd))], [element[1] for element in theseXYdata if (element[0]>xAxisMean-(2.5*xAxisStd) and element[0]<xAxisMean+(2.5*xAxisStd))])
				# 		
				# 		pl.scatter(averagesPerBinXaxis,predictedAveragesPerBinYaxis,marker='x',color='k',s=70)
				# 		
				# 		pl.scatter(averagesPerBinXaxis, averagesPerBinYaxis,color='r')
				# 		
				# 		pl.xlabel(['x','y'][xyIndex]+' position')
				# 		pl.ylabel('Pupil size')
				# 		
				# 		#---------
				# 		
				# 		theseXYdataCorrected=[[element[0],element[1]-mySurface(element[0],element[2],theseCoeffs[0],theseCoeffs[1],otherCoeffs[1],theseCoeffs[2],otherCoeffs[2],theseCoeffs[3],otherCoeffs[3])] for element in theseXYdata]
				# 		theseXYdataCorrected.sort()
				# 		
				# 		averagesPerBinYaxis=[]
				# 		for binBorderIndex in range(len(binBorders)-1):
				# 			averagesPerBinYaxis=averagesPerBinYaxis+[numpy.average([theseXYdataCorrected[thisIndex][1] for thisIndex in range(binBorders[binBorderIndex],binBorders[binBorderIndex+1])])]
				# 	
				# 		pl.figure(f_corrected.number)
				# 		s=f_corrected.add_subplot(2,2,xyIndex+1+leftRightIndex*2)
				# 		
				# 		s.set_title(['Left eye, corrected', 'Right eye, corrected'][leftRightIndex])
				# 		
				# 		#are still the same anyway
				# 		#xAxisMean=numpy.average([element[0] for element in theseXYdataCorrected])
				# 		#xAxisStd=numpy.std([element[0] for element in theseXYdataCorrected])
				# 		
				# 		pl.scatter([element[0] for element in theseXYdataCorrected if (element[0]>xAxisMean-(2.5*xAxisStd) and element[0]<xAxisMean+(2.5*xAxisStd))], [element[1] for element in theseXYdataCorrected if (element[0]>xAxisMean-(2.5*xAxisStd) and element[0]<xAxisMean+(2.5*xAxisStd))])
				# 		pl.scatter(averagesPerBinXaxis, averagesPerBinYaxis,color='r')
				# 		
				# 		pl.xlabel(['x','y'][xyIndex]+' position')
				# 		pl.ylabel('Pupil size, corrected')
				# 		
				# pl.figure(f.number)	
				# pl.savefig(myPath+figuresSubFolder+'/forshortening_preCorrection_'+thisEDFfileWithoutEdf+'.png')
				# pl.close()
				# 				
				# pl.figure(f_corrected.number)	
				# pl.savefig(myPath+figuresSubFolder+'/forshortening_postCorrection_'+thisEDFfileWithoutEdf+'.png')
				# pl.close()
				
				#coeffs have been stored; will be incorporated in the following lines, which are there anyway in the
				#context of concatenating the two files of the same condition
				#---------------	
				
				#--------------------------------
				#if the sampling rate was accidentally set to 2000 instead of 1000: take every alternate sample
				
				f = open(myPath+eyeSubFolder+'/'+beforeMergingFolder+'/'+thisEDFfileWithoutEdf+'_e.asc', 'r')
				rawtextEventData=f.read()
				f.close()
				
				regExp='RECORD CR ([0-9]*) '
				samplingRateInfo=list(set(re.findall(regExp,rawtextEventData)).union())

				if len(samplingRateInfo)>1:

					userinteraction.printFeedbackMessage('Tracker sampling rate was changed during the experiment. Selectively downsampling.')
					regExp='([0-9]*) !MODE RECORD CR ([0-9]*) '
					samplingRateInfoPlusTime=re.findall(regExp,rawtextEventData)
					samplingRateInfoPlusTime=samplingRateInfoPlusTime+[(str(float(newSampleData[-1][0])+1.),'yomoma')]

					sampleDataSelectivelyDownsampled=[]
					for oneAdjustIndex in range(len(samplingRateInfoPlusTime)-1):

						startTime=float(samplingRateInfoPlusTime[oneAdjustIndex][0])
						endTime=float(samplingRateInfoPlusTime[oneAdjustIndex+1][0])

						startIndex=[index for index in range(len(newSampleData)) if float(newSampleData[index][0])==startTime][0]
						endIndex=[index for index in range(len(newSampleData)) if float(newSampleData[index][0])<endTime][-1]

						if samplingRateInfoPlusTime[oneAdjustIndex][1]=='1000':
							sampleDataSelectivelyDownsampled=sampleDataSelectivelyDownsampled+newSampleData[startIndex:(endIndex+1)]
						elif samplingRateInfoPlusTime[oneAdjustIndex][1]=='2000':
							sampleDataSelectivelyDownsampled=sampleDataSelectivelyDownsampled+[newSampleData[index] for index in range(startIndex,endIndex,2)]	#instead of taking the average between these two values recorded at 2000 Hz, we're going to choose the easy option of just taking the first value
						else:
							raise Exception('BS sampling rate. Forget it.')
					newSampleData=sampleDataSelectivelyDownsampled

				elif samplingRateInfo[0]=='1000':
					userinteraction.printFeedbackMessage('Sampling rate 1000 Hz. Everything good.')
				elif samplingRateInfo[0]=='2000':
					userinteraction.printFeedbackMessage('Sampling rate 2000 Hz. Downsampling.')
					newSampleData=[newSampleData[index] for index in range(0,len(newSampleData)-1,2)]	#instead of taking the average between these two values recorded at 2000 Hz, we're going to choose the easy option of just taking the first value
				else:
					raise Exception('BS sampling rate. Forget it.')
				#--------------------------------
				
				newDataTrialStartPoss=[index for index in range(len(newEventData)) if re.findall('([0-9]*) *Trial ([0-9]*) started',newEventData[index][-1])]
				
				if not counter==0:
					
					oldDataLastTrialTime=newDataLastTrialTime
					newDataFirstTrialTime=newEventData[newDataTrialStartPoss[0]][1]
					
					oldDataLastTrialTimeInt=int(re.findall('([0-9]*) *Trial [0-9]* started',oldDataLastTrialTime)[0])
					newDataFirstTrialTimeInt=int(re.findall('([0-9]*) *Trial [0-9]* started',newDataFirstTrialTime)[0])
					
					actualTimeGap_ms=newDataFirstTrialTimeInt-oldDataLastTrialTimeInt
					timeCorrectionRequired=1000*simulatedTimeGapForMerging_s-actualTimeGap_ms
					
					newDataLastTrialTime=newEventData[newDataTrialStartPoss[-1]][1]
					
					newSampleData=[[str(int(element[0])+timeCorrectionRequired)]+element[1:] for element in newSampleData]
					
					# allWithinTrialIndicesWithLegalLeftEye=[candidateIndex for candidateIndex in allWithinTrialIndices if not(float(newSampleData[candidateIndex][3])==0)]
					# allWithinTrialIndicesWithLegalRightEye=[candidateIndex for candidateIndex in allWithinTrialIndices if not(float(newSampleData[candidateIndex][6])==0)]
					# 
					# coefficientsLeftEye=coefficientsPerEye[0]
					# for replacementIndex in allWithinTrialIndicesWithLegalLeftEye:
					# 	thisX=float(newSampleData[replacementIndex][1])
					# 	thisY=float(newSampleData[replacementIndex][2])
					# 	thisUncorrectedPupil=float(newSampleData[replacementIndex][3])
					# 	correctedPupil=thisUncorrectedPupil-mySurface(thisX,thisY,coefficientsLeftEye[0],coefficientsLeftEye[1],coefficientsLeftEye[2],coefficientsLeftEye[3],coefficientsLeftEye[4],coefficientsLeftEye[5],coefficientsLeftEye[6])
					# 	newSampleData[replacementIndex][3]=str(correctedPupil)
					# 
					# coefficientsRightEye=coefficientsPerEye[1]
					# for replacementIndex in allWithinTrialIndicesWithLegalRightEye:
					# 	thisX=float(newSampleData[replacementIndex][4])
					# 	thisY=float(newSampleData[replacementIndex][5])
					# 	thisUncorrectedPupil=float(newSampleData[replacementIndex][6])
					# 	correctedPupil=thisUncorrectedPupil-mySurface(thisX,thisY,coefficientsRightEye[0],coefficientsRightEye[1],coefficientsRightEye[2],coefficientsRightEye[3],coefficientsRightEye[4],coefficientsRightEye[5],coefficientsRightEye[6])
					# 	newSampleData[replacementIndex][6]=str(correctedPupil)
					
					firstIndexToBeIncluded=[lineIndex for lineIndex,lineValue in enumerate(newEventData) if lineValue[0]=='INPUT'][0]
					newEventData=newEventData[firstIndexToBeIncluded:]
					
					correctedEventData=[]
					for oneEventLine in newEventData:
						
						if oneEventLine[0] in ['INPUT','MSG','START','END']:
							
							secondColumn=oneEventLine[1]
							secondColumnParsed=re.findall('([0-9]*)(.*)',secondColumn)[0]
							oneEventLineAdjusted=[oneEventLine[0],str(int(secondColumnParsed[0])+timeCorrectionRequired)+secondColumnParsed[1]]+oneEventLine[2:]
							
						elif re.findall('(SFIX|SSACC|SBLINK).*([0-9]*)',oneEventLine[0]):
							
							firstColumn=oneEventLine[0]
							firstColumnParsed=re.findall('(SFIX|SSACC|SBLINK)(\s*[LR]\s*)([0-9]*)',firstColumn)[0]
							oneEventLineAdjusted=[firstColumnParsed[0]+firstColumnParsed[1]+str(int(firstColumnParsed[2])+timeCorrectionRequired)]+oneEventLine[1:]
						
						elif re.findall('(EFIX|ESACC|EBLINK).*([0-9]*)',oneEventLine[0]):
							
							firstColumn=oneEventLine[0]
							firstColumnParsed=re.findall('(EFIX|ESACC|EBLINK)(\s*[LR]\s*)([0-9]*)',firstColumn)[0]
							
							secondColumn=oneEventLine[1]
							
							oneEventLineAdjusted=[firstColumnParsed[0]+firstColumnParsed[1]+str(int(firstColumnParsed[2])+timeCorrectionRequired)]+[str(int(secondColumn)+timeCorrectionRequired)]+oneEventLine[2:]
							
						else:
							
							oneEventLineAdjusted=oneEventLine
						
						correctedEventData=correctedEventData+[oneEventLineAdjusted]
						
					newEventData=correctedEventData

				else:

					newDataLastTrialTime=newEventData[newDataTrialStartPoss[-1]][1]
					
					# allWithinTrialIndicesWithLegalLeftEye=[candidateIndex for candidateIndex in allWithinTrialIndices if not(float(newSampleData[candidateIndex][3])==0)]
					# allWithinTrialIndicesWithLegalRightEye=[candidateIndex for candidateIndex in allWithinTrialIndices if not(float(newSampleData[candidateIndex][6])==0)]
					# 
					# coefficientsLeftEye=coefficientsPerEye[0]
					# for replacementIndex in allWithinTrialIndicesWithLegalLeftEye:
					# 	thisX=float(newSampleData[replacementIndex][1])
					# 	thisY=float(newSampleData[replacementIndex][2])
					# 	thisUncorrectedPupil=float(newSampleData[replacementIndex][3])
					# 	correctedPupil=thisUncorrectedPupil-mySurface(thisX,thisY,coefficientsLeftEye[0],coefficientsLeftEye[1],coefficientsLeftEye[2],coefficientsLeftEye[3],coefficientsLeftEye[4],coefficientsLeftEye[5],coefficientsLeftEye[6])
					# 	newSampleData[replacementIndex][3]=str(correctedPupil)
					# 
					# coefficientsRightEye=coefficientsPerEye[1]
					# for replacementIndex in allWithinTrialIndicesWithLegalRightEye:
					# 	thisX=float(newSampleData[replacementIndex][4])
					# 	thisY=float(newSampleData[replacementIndex][5])
					# 	thisUncorrectedPupil=float(newSampleData[replacementIndex][6])
					# 	correctedPupil=thisUncorrectedPupil-mySurface(thisX,thisY,coefficientsRightEye[0],coefficientsRightEye[1],coefficientsRightEye[2],coefficientsRightEye[3],coefficientsRightEye[4],coefficientsRightEye[5],coefficientsRightEye[6])
					# 	newSampleData[replacementIndex][6]=str(correctedPupil)
					
				if not counter==(len(oneSetOfNumbers)-1):
					lastIndexToBeIncluded=-4
					newEventData=newEventData[:lastIndexToBeIncluded+1]
				
				allSampleData=allSampleData+newSampleData
				allEventData=allEventData+newEventData
			
			with open(myPath+eyeSubFolder+'/'+thisSampleOutName, 'w') as f:
				for line in allSampleData:
					outline='\t'.join(line)+'\n'
					f.write(outline)
			f.close()
				
			with open(myPath+eyeSubFolder+'/'+thisEventOutName, 'w') as f:
				for line in allEventData:
					outline='\t'.join(line)+'\n'
					f.write(outline)
			f.close()
			
		else:
			
			print 'not merging eye files '+thisSampleOutName+' and '+thisEventOutName+' because already done'

#-----------

allFilenamesBehavioral=os.listdir(myPath+behavioralSubFolder)
outputFilenames=[element for element in allFilenamesBehavioral if 'observer' in element]
allFilenamesEye=os.listdir(myPath+eyeSubFolder)
outputFilenamesEye=[element[:-6] for element in allFilenamesEye if re.findall('[a-zA-Z0-9]*C[0-9]*_[0-9]',element) and ('s.asc' in element)]

# --------sort through behavioral data---------
data=[{'fileName':fileName,'data':numpy.load(myPath+behavioralSubFolder+'/'+fileName,allow_pickle=True),'observer':fileName.split('_')[2],'session':fileName.split('_')[4]} for fileName in outputFilenames]

uniqueObservers=list(set([element['observer'] for element in data]).union())

trialInfo=[]
for oneObserver in uniqueObservers:
	
	thisInfo=[element for element in data if element['observer']==oneObserver]
	
	for sessionIndex,oneSession in enumerate(thisInfo):
		
		thisSessionNumber=oneSession['session']
		
		thisEyeFilenameCandidates=[oneFile for oneFile in outputFilenamesEye if oneObserver+'C'+thisSessionNumber+'_' in oneFile]

		if thisEyeFilenameCandidates:
			thisEyeFilename=thisEyeFilenameCandidates[0]		#in this case, because of the file merging operation above, this is the name of a fictional edf file that would have lay at the basis of the merged .asc files
		else:
			print 'no eye file for observer '+oneObserver+', session: '+thisSessionNumber
			thisEyeFilename='missing'
		
		theseData=oneSession['data']
		trialStartPoss=[index for index in range(len(theseData)) if theseData[index][0]==trialStartCode]
		
		trialNumber=0
		for trialStartPosIndex in range(0,len(trialStartPoss),2):	#every second 'trialStartCode' line is an actual one; every first of a pair is actually more like the start of drift correction
			
			driftCorrStartTime=theseData[trialStartPoss[trialStartPosIndex]][1]
			trialStartTime=theseData[trialStartPoss[trialStartPosIndex+1]][1]
			trialMotionDirections=theseData[trialStartPoss[trialStartPosIndex]][2]
			trialColorProp=colorProps[theseData[trialStartPoss[trialStartPosIndex+1]][4]]
			
			trialEndTime=trialStartTime+trialDurS
			driftCorrEndTime=trialStartTime-waitDurS		#this can also include calibration if we recalibrated, in which case eye tracking has been off for a while
			
			try:
				allEventsThisTrial=theseData[trialStartPoss[trialStartPosIndex]+2:trialStartPoss[trialStartPosIndex+2]]
			except IndexError:
				allEventsThisTrial=theseData[trialStartPoss[trialStartPosIndex]+2:]
			
			theseDirChangeReports=[[element[1],element[2]] for element in allEventsThisTrial if element[0]==dirChangeReportedCode]	#time, which was it?
			theseSizeProbes=[[element[1],element[3]] for element in allEventsThisTrial if element[0]==detectionDotCode]
			thesePhysDirChanges=[[element[1],element[2]] for element in allEventsThisTrial if element[0]==dirChangedCode]
			theseProbeReports=[[element[1],element[2]] for element in allEventsThisTrial if element[0]==keyPressedCode]	#time, hit (1) or FA (-1)
			
			#firstChangeIdentity=theseDirChanges[0][1]
			#theseDirChanges=[[trialStartTime,abs(firstChangeIdentity-1.)]]+theseDirChanges
			
			trialInfo=trialInfo+[{'trialNumber':trialStartPosIndex,'observer':oneObserver,'session':thisSessionNumber,'absoluteTrialStartTime':trialStartTime,'trialEndTime':trialEndTime-trialStartTime,'driftCorrStartTime':driftCorrStartTime-trialStartTime,'directions':trialMotionDirections, 'colorProp':trialColorProp,'dirChangeReports':[[element[0]-trialStartTime, element[1]] for element in theseDirChangeReports],'sizeProbes':[[element[0]-trialStartTime, element[1]] for element in theseSizeProbes],'dirChangesPhys':[[element[0]-trialStartTime, element[1]] for element in thesePhysDirChanges],'probeReports':[[element[0]-trialStartTime, element[1]] for element in theseProbeReports],'edfFile':thisEyeFilename}]

			trialNumber=trialNumber+1
#----------------------

#-----do eye stuff-----

#preprocessing analysis settings
filterCutoffs_Hz=[.01,6.]#altered 6/20/19 based on Tomas' recommendation. [0.05,4.]	#for 3rd order Butterworth		#only low-pass used! High-pass replaced with expo fit.
blinkFlankDurInterpolation_s=.050		#how large a window on the side of a blink to take to constrain linear interpolation
blinkSideBufferPre_s=.050	#how long before the blink to discard data
blinkSideBufferPost_s=.085	#how long before the blink to discard data
timePerRowAscFile_s=0.001	#how many s per row in the sample file
numColumsExpected='notused'	#sometimes the edf file, for some weird reason, doesn't have the right number of columns because, apparently, some data wasn't recorded. In those case, simply skip this scan.
#maxBlinkDur_s=1.0	#longer than this is not currently also interpolated; not deleted (it's not actually the duration of the missing data but the duration of the missing data + 2x blinkSideBuffer_s)
#NOTE: All blinks and signal drops, regardless of duration, are interpolated and written as events to event file ending in .txt for future regression.
#Then later maxBlinkDur_s is used again to decide whether to actually include in regressor by adding into infoDict. Kind of nonsense to do it this way, but at least the .txt file has all blinks now for reference, including ones that are too long to be actual blinks.
minChunkDur_s=2.0	#chunks of data flanked by missing data should be at least this long to be included
timeZeroInfo=['MSG\t *([0-9]*) *Trial 0 started',0,2.]	#[regexp into event file, rank number of hits to that regexp,seconds to spare in front]: the first two elements together define the event that is time zero; the last one indicates how many seconds to spare in front of that moment, to avoid filter edge artefacts and stuff
microsaccVelThresh=6	#unit is median-based standard-deviations, I believe
microsaccMinDur_ms=6. 	#minimum number of ms for something to be a saccade

preprocessInfoDict={'filterCutoffs_Hz':filterCutoffs_Hz,'blinkFlankDurInterpolation_s':blinkFlankDurInterpolation_s,
					'blinkSideBufferPre_s':blinkSideBufferPre_s,'blinkSideBufferPost_s':blinkSideBufferPost_s,'numColumsExpected':numColumsExpected,'timePerRowAscFile_s':timePerRowAscFile_s,
					'minChunkDur_s':minChunkDur_s,'timeZeroInfo':timeZeroInfo,'microsaccVelThresh':microsaccVelThresh,'microsaccMinDur_ms':microsaccMinDur_ms}
					
timecourseDataSubFolder='filtered'
regressorsSubFolder='pupilRegressors'
GLMoutcomeSubFolder='GLMoutcomes'

for oneObserver in uniqueObservers:
	
	#dirChangeToColor1EventRegressor=[]#1 for occurrence; 0 for no occurrence
	#dirChangeToColor2EventRegressor=[]#1 for occurrence; 0 for no occurrence
	
	dataThisObserver=[element for element in trialInfo if element['observer']==oneObserver]
	
	uniqueSessions=list(set([element['session'] for element in dataThisObserver]).union())
	uniqueSessions=[int(element) for element in uniqueSessions]
	uniqueSessions.sort()
	uniqueSessions=[str(element) for element in uniqueSessions]
	
	for oneSession in uniqueSessions:
		
		trialOnsetRegressor=[]#1 for occurence
		trialOffsetRegressor=[]#1 for occurrence
		dirChangeReportEventRegressor=[]#1 for occurrence; 0 for no occurrence; new dir
		dirChangePhysEventRegressor=[]#1 for occurrence; 0 for no occurrence; new dir
		probeEventRegressor=[]#1 for occurrence; 0 for no occurrence; size gain
		probeReportEventRegressor=[]#1 for occurrence; 0 for no occurrence; hit or miss
		
		dataThisSession=[element for element in dataThisObserver if element['session']==oneSession]
		sessionStartTime=[element['absoluteTrialStartTime'] for element in dataThisSession if element['trialNumber']==0][0]

		thisEDFfile=dataThisSession[0]['edfFile']
		
		if thisEDFfile=='missing':
			print 'There is no edf file so not doing eye stuff for observer: '+oneObserver+', session: '+oneSession
		else:
			thisEDFfileWithoutEdf=thisEDFfile
			
			#because of the merge operation at the top, the ascs are already there
			
			# if '.edf' in thisEDFfile:
			# 	thisEDFfileWithoutEdf=thisEDFfile[:-4]
			# else:
			# 	thisEDFfileWithoutEdf=thisEDFfile
			# 	commandline.ShellCmd('mv '+myPath+thisEDFfile+' '+myPath+thisEDFfile+'.edf')
			# 	
			# if thisEDFfile+'_s.asc' in allFilenamesEye:
			# 	print 'EDF2ASC already done for observer '+oneObserver+', session '+oneSession
			# else:
			# 		
			# 	for eventOrSample in ['e','s']:
			# 		commandline.ShellCmd('edf2asc -'+eventOrSample+' '+myPath+eyeSubFolder+'/'+thisEDFfileWithoutEdf+'.edf')
			# 		commandline.ShellCmd('mv '+myPath+eyeSubFolder+'/'+thisEDFfileWithoutEdf+'.asc '+myPath+eyeSubFolder+'/'+thisEDFfileWithoutEdf+'_'+str(eventOrSample)+'.asc')
			# 	commandline.ShellCmd('mv '+myPath+eyeSubFolder+'/'+thisEDFfileWithoutEdf+'.edf '+myPath+eyeSubFolder+'/'+thisEDFfileWithoutEdf)
		
		outputFilename='observer'+oneObserver+'session'+oneSession

		analysis.doNotFilterButDoCleanPupilSizeAfterTomasAndGillesInputSept2020(preprocessInfoDict,myPath+eyeSubFolder+'/',myPath+timecourseDataSubFolder+'/',myPath+regressorsSubFolder+'/',thisEDFfileWithoutEdf,outputFilename)	#process and store pupil stuff THIS ONLY DETECTS AND INTERPOLATES BLINKS AND OTHER SIGNAL DROPS (INCLUDING RECALIBRATION); NOTHING ELSE
		
		analysis.detectSaccades(preprocessInfoDict,myPath+eyeSubFolder+'/',myPath+timecourseDataSubFolder+'/',myPath+regressorsSubFolder+'/',thisEDFfileWithoutEdf,outputFilename)	#process and store saccade stuff
	
		#vvvvv create and store regressors for non-eye events vvvv
		
		regressorFilesPresent=commandline.putFileNamesInArray(myPath+regressorsSubFolder+'/')
		if not(outputFilename+'_trialOnset.txt' in regressorFilesPresent):
			
			userinteraction.printFeedbackMessage("creating and storing behavioral regressors for "+outputFilename)
			for index in range(len(dataThisSession)):
				dataThisSession[index]['relativeTrialStartTime']=dataThisSession[index]['absoluteTrialStartTime']-sessionStartTime
		
			for oneTrial in dataThisSession:
				temporalOffsetThisTrial=oneTrial['relativeTrialStartTime']
				dirChangeReportsThisTrial=oneTrial['dirChangeReports']
				dirChangePhysEventsThisTrial=oneTrial['dirChangesPhys']
				probeEventsThisTrial=oneTrial['sizeProbes']
				colorPropThisTrial=oneTrial['colorProp']
				probeReportsThisTrial=oneTrial['probeReports']
				
				trialOnsetRegressor=trialOnsetRegressor+[[temporalOffsetThisTrial,1,oneTrial['directions'][0],oneTrial['directions'][1]]]
				trialOffsetRegressor=trialOffsetRegressor+[[temporalOffsetThisTrial+trialDurS,1]]
				dirChangeReportEventRegressor=dirChangeReportEventRegressor+[[element[0]+temporalOffsetThisTrial,1.,element[1]] for element in dirChangeReportsThisTrial]
				dirChangePhysEventRegressor=dirChangePhysEventRegressor+[[element[0]+temporalOffsetThisTrial,1.,element[1]] for element in dirChangePhysEventsThisTrial]
				probeEventRegressor=probeEventRegressor+[[element[0]+temporalOffsetThisTrial,1.,element[1]] for element in probeEventsThisTrial]
				probeReportEventRegressor=probeReportEventRegressor+[[element[0]+temporalOffsetThisTrial,1.,element[1]] for element in probeReportsThisTrial]
				
				# if colorPropThisTrial==1.:
				# 	dirChangeToColor1EventRegressor=dirChangeToColor1EventRegressor+[[element[0]+temporalOffsetThisTrial,1.] for element in dirChangesThisTrial if element[1]==0]
				# 	dirChangeToColor2EventRegressor=dirChangeToColor2EventRegressor+[[element[0]+temporalOffsetThisTrial,1.] for element in dirChangesThisTrial if element[1]==1]
				# elif colorPropThisTrial==0.:
				# 	dirChangeToColor1EventRegressor=dirChangeToColor1EventRegressor+[[element[0]+temporalOffsetThisTrial,1.] for element in dirChangesThisTrial if element[1]==1]
				# 	dirChangeToColor2EventRegressor=dirChangeToColor2EventRegressor+[[element[0]+temporalOffsetThisTrial,1.] for element in dirChangesThisTrial if element[1]==0]
				# else:
				# 	print "GOT UNEXPECTED COLOR PROPORTIONS! EXPECTED THEM TO BE EITHER 0. OR 1. WHAT's GOING ON?"
					
			# allRegressors=[trialOnsetRegressor,trialOffsetRegressor,dirChangeEventRegressor,dirChangeSizeRegressor,speedBumpRegressor,speedBumpSizeRegressor,speedBumpDetectedRegressor,dirChangeColorStepRegressor]
			# allRegressorNames=['trialOnset','trialOffset','dirChangeEvent','dirChangeSize','speedBump','speedBumpSize','speedBumpDetected','colorStepSize','dirChangeIncColorStep','dirChangeExcColorStep']
		
			allRegressors=[trialOnsetRegressor,trialOffsetRegressor,dirChangeReportEventRegressor,dirChangePhysEventRegressor,probeEventRegressor,probeReportEventRegressor]
			allRegressorNames=['trialOnset','trialOffset','dirChangeReportEvent','dirChangePhysEvent','probeEvent','probeReportEvent']

			for oneRegressorIndex in range(len(allRegressors)):
				if allRegressors[oneRegressorIndex]:	#skip if empty
					if allRegressorNames[oneRegressorIndex]=='trialOnset':
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f\t%20.10f\t%20.10f')
					elif  allRegressorNames[oneRegressorIndex]=='dirChangeReportEvent':
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f\t%20.10f')
					elif  allRegressorNames[oneRegressorIndex]=='dirChangePhysEvent':
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f\t%20.10f')
					elif  allRegressorNames[oneRegressorIndex]=='probeEvent':
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f\t%20.10f')
					elif  allRegressorNames[oneRegressorIndex]=='probeReportEvent':
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f\t%20.10f')
					else:
						numpy.savetxt(myPath+regressorsSubFolder+'/'+outputFilename+'_'+allRegressorNames[oneRegressorIndex]+'.txt',allRegressors[oneRegressorIndex],fmt='%20.10f\t%20.10f')
					
		else:
			
			userinteraction.printFeedbackMessage("not creating behavioral regressors for "+outputFilename+" because already done")
			
			#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		analysis.reviewPlots(myPath+timecourseDataSubFolder+'/',myPath+regressorsSubFolder+'/',myPath+figuresSubFolder+'/',outputFilename,5.)
		
		analysis.isolateAndNormalizePursuitComponent(myPath+timecourseDataSubFolder+'/',myPath+miscStuffSubFolder+'/',myPath+regressorsSubFolder+'/',myPath+figuresSubFolder+'/',outputFilename,.02,.05,[.25,.4,.1],.75,10.)

categoryLimitsSwitchIdentification=[-.85,.85]	#altered Nov 14. for identifying percept, same in other direction
analysis.identifySwitches(myPath+timecourseDataSubFolder+'/',myPath+regressorsSubFolder+'/',myPath+figuresSubFolder+'/',myPath+behavioralSubFolder+'/',myPath+miscStuffSubFolder+'/',categoryLimitsSwitchIdentification,10000)

analysis.plotReportedSwitchInfo(myPath+regressorsSubFolder+'/',myPath+figuresSubFolder+'/',myPath+behavioralSubFolder+'/')

#Tidy up and assemble regressors. Regress ook blinks en saccs uit. En bepaal met deconvolutie de temporele relatie tussen inferred, physical, reported. Beetje rommelig maar ja:
#This is also the stage in which only events that are far enough from missing data for the deco-window not to touch it, are stored in allInfoDictList to be analyzed further. Up until this point all events are stored regardless of that.

maxBlinkDur_s=.9 		#this max dur and the min dur are used so that fewer signal drops are treated as blinks when it comes to computing (and regressing out) the blink-related pupil response
minBlinkDur_s=.13		#see Kwon, K., Shipley, R., Edirisinghe, M., Ezra, D., Rose, G., Best, S., Cameron, R. (2013). High-speed camera characterization of voluntary eye blinking kinematics Journal of The Royal Society Interface  10(85), 20130227. https://dx.doi.org/10.1098/rsif.2013.0227
minInterSaccInterval=.1	#everything closer together than this seems to be just a bunch of 'saccades' as part of a square wave jerk, and I don't want to actually count those as like 2 or 3 saccades
basicSampleRate=1000	#Hz, of the recorded data in the pupil file
downSampleRate=100
downSampleRateBehavioralSwitchToSwitch=100
decoInterval=[-3.5,6.5]	#for pupil data; not behavioral data.
decoIntervalSacc=[-.5,4.5]
decoIntervalBlink=[-.5,7.5]
print('make sure deco intervals match those used in later code that runs GLMS! Here they are only used to remove events that are too close to trial edge.')

decoInterval_switches_riv_inf_rep=[-2.,1.]	#for behavioral data. name is condition_y_x, so this one is: rivalry, inferred, reported
decoInterval_switches_repl_inf_phys=[-1.,2.]	#for behavioral data.
decoInterval_switches_repl_inf_rep=[-1.5,1.5]	#for behavioral data.
decoInterval_switches_repl_rep_phys=[-1.5,1.5]	#for behavioral data.
ridgeForrester=False
plotColors=['r','g','b']
probeResponseWindow_s=2.	#probes not followed by a spacebar within this interval are considered not detected

newSampleRate=basicSampleRate/downSampleRate

allRegressorFileNames=[element for element in os.listdir(myPath+regressorsSubFolder) if 'observer' in element]
allBehavioralFileNames=[element for element in os.listdir(myPath+behavioralSubFolder) if 'observer' in element]
allFilteredEyeFileNames=[element for element in os.listdir(myPath+timecourseDataSubFolder) if 'observer' in element]

allObs=[re.findall('observer([a-zA-Z0-9]*)session',element) for element in allRegressorFileNames]
allObs=[element[0] for element in allObs if element]
uniqueObservers=list(set(allObs).union())

allSess=[re.findall('observer[a-zA-Z0-9]*session([0-9]*)',element) for element in allRegressorFileNames]
allSess=[element[0] for element in allSess if element]
uniqueSess=list(set(allSess).union())

allInfoDictListFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_allInfoDictList' in element]
allInfoDictList=[]

for oneObs in uniqueObservers:
	
	if oneObs+'_allInfoDictList' in allInfoDictListFilesPresent:
		
		print 'Reading allInfoDictList from file for observer '+oneObs+' because already present.'
	
		with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', "r") as f:
			allInfoDictListThisObs=pickle.load(f)
			
	else:
			
		allInfoDictListThisObs=[]
		for oneSess in uniqueSess:
		
			thisDict={'obs':oneObs,'sess':oneSess}
			
			#--------
			#gather all events
			
			pupilDataFile=[element for element in allFilteredEyeFileNames if 'observer'+oneObs+'session'+oneSess in element and '_cleaned_pup_GT0920' in element][0]
			pupilData=filemanipulation.readDelimitedIntoArray(myPath+timecourseDataSubFolder+'/'+pupilDataFile,'\t')
			startTimePupilRecording_s=pupilData[0][0]/1000 	#pupil data recording usually starts a bit before the start of the first trial, event times have been stored relative to start of first trial, and firdeconv interprets event times as relative to start of pupil time series. So this starting point will be used to move the event times so that they become relative to recording onset rather than the start of the first trial
		
			blinkDataFile=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'blink_moments_and_durations' in element][0]
			blinkData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+blinkDataFile,' ')
			blinkData=[[float(element[0]),float(element[-1])] for element in blinkData]
			blinkMoments_s=[element[0]-startTimePupilRecording_s for element in blinkData if element[1]<=maxBlinkDur_s and element[1]>=minBlinkDur_s]
			thisDict['blinks']=numpy.array(blinkMoments_s)		#blink regressors are set at the start of the blink
			
			signalDropStartsAndEnds_s=[[element[0]-startTimePupilRecording_s,(element[0]+element[1])-startTimePupilRecording_s] for element in blinkData if element[1]>maxBlinkDur_s or element[1]<minBlinkDur_s]
			thisDict['signalDropStarts']=numpy.array([element[0] for element in signalDropStartsAndEnds_s])
			thisDict['signalDropEnds']=numpy.array([element[1] for element in signalDropStartsAndEnds_s])
		
			saccadeDataFile=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'binocularsaccadeMomentsStandardRegressorFormat' in element][0]
			saccadeData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+saccadeDataFile,'\t')
			saccadeMoments_s=[float(element[0])-startTimePupilRecording_s for element in saccadeData]
			
			#----THIS IS FOR MERGING SACCADES THAT ARE TOO CLOSE TOGETHER
			interSaccIntervals=numpy.array(saccadeMoments_s[1:])-numpy.array(saccadeMoments_s[:len(saccadeMoments_s)-1])
			
			mergeIndices=[intervalIndex+1 for intervalIndex,intervalDur in enumerate(interSaccIntervals) if intervalDur<minInterSaccInterval]
			nonMergeIndices=[intervalIndex+1 for intervalIndex,intervalDur in enumerate(interSaccIntervals) if intervalDur>=minInterSaccInterval]
			
			if len(mergeIndices)>0:
				groupedMergeIndices=[]
				dropFromNonMergeIndices=[]
				currentGroup=[mergeIndices[0]]
				for mergeIndex in mergeIndices[1:]:
					if mergeIndex-currentGroup[-1]==1:
						currentGroup=currentGroup+[mergeIndex]
					else:
						startOfGroupedBlock=[currentGroup[0]-1]
						dropFromNonMergeIndices=dropFromNonMergeIndices+startOfGroupedBlock
						groupedMergeIndices=groupedMergeIndices+[startOfGroupedBlock+currentGroup]
						currentGroup=[mergeIndex]
						
				startOfGroupedBlock=[currentGroup[0]-1]
				dropFromNonMergeIndices=dropFromNonMergeIndices+startOfGroupedBlock
				groupedMergeIndices=groupedMergeIndices+[startOfGroupedBlock+currentGroup]
				nonMergeIndices=[thisCandidate for thisCandidate in nonMergeIndices if not thisCandidate in dropFromNonMergeIndices]
				
				saccadeMomentsIndividuals_s=[saccadeMoments_s[thisIndex] for thisIndex in nonMergeIndices]
				saccadeMomentsGrouped_s=[numpy.average([saccadeMoments_s[thisIndex] for thisIndex in thisGroup]) for thisGroup in groupedMergeIndices]
				
				oldTotal=len(saccadeMoments_s)
				
				saccadeMoments_s=saccadeMomentsIndividuals_s+saccadeMomentsGrouped_s
				saccadeMoments_s.sort()
				
				print 'Removing '+str(oldTotal-len(saccadeMoments_s))+' out of '+str(oldTotal)+' saccades for observer '+oneObs+', session '+str(oneSess)+' because too close together.'
			#--------------------------END
			
			thisDict['saccades']=numpy.array(saccadeMoments_s)
		
			trialStartDataFile=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'trialOnset' in element][0]
			trialStartData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+trialStartDataFile,'\t')
			trialStartMoments_s=[float(element[0])-startTimePupilRecording_s for element in trialStartData]
			thisDict['trialStarts']=numpy.array(trialStartMoments_s)
		
			trialEndDataFile=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'trialOffset' in element][0]
			trialEndData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+trialEndDataFile,'\t')
			trialEndMoments_s=[float(element[0])-startTimePupilRecording_s for element in trialEndData]
			thisDict['trialEnds']=numpy.array(trialEndMoments_s)
			
			pupilDataFileNonInterpolated=[element for element in allFilteredEyeFileNames if 'observer'+oneObs+'session'+oneSess in element and 'non_interpolated_pupil' in element][0]
			pupilDataNonInterpolated=filemanipulation.readDelimitedIntoArray(myPath+timecourseDataSubFolder+'/'+pupilDataFileNonInterpolated,'\t')
			thisDict['almostRawPupilSamples']=numpy.array([element[1] for element in pupilDataNonInterpolated])

			gazeXFile=[element for element in allFilteredEyeFileNames if 'observer'+oneObs+'session'+oneSess in element and 'xGaze_on_pupil_axis' in element][0]
			gazeXData=filemanipulation.readDelimitedIntoArray(myPath+timecourseDataSubFolder+'/'+gazeXFile,'\t')
			thisDict['gazeXSamples']=numpy.array([element[1] for element in gazeXData])
			
			gazeYFile=[element for element in allFilteredEyeFileNames if 'observer'+oneObs+'session'+oneSess in element and 'yGaze_on_pupil_axis' in element][0]
			gazeYData=filemanipulation.readDelimitedIntoArray(myPath+timecourseDataSubFolder+'/'+gazeYFile,'\t')
			thisDict['gazeYSamples']=numpy.array([element[1] for element in gazeYData])
			
			
			reportedProbeDataFileCandidates=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'probeReportEvent' in element]
			if reportedProbeDataFileCandidates:
				reportedProbeDataFile=reportedProbeDataFileCandidates[0]
				reportedProbeData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+reportedProbeDataFile,'\t')
				reportedProbeMoments_s=[float(element[0])-startTimePupilRecording_s for element in reportedProbeData]
				thisDict['probeReports']=numpy.array(reportedProbeMoments_s)
			else:
				reportedProbeMoments_s=[]
				thisDict['probeReports']=numpy.array([])		#this is time-aligned to the key press moments, irrespective of when (and, in fact, wether) a probe occurred

			shownProbeDataFileCandidates=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'probeEvent' in element]
			if shownProbeDataFileCandidates:
				shownProbeDataFile=shownProbeDataFileCandidates[0]
				shownProbeData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+shownProbeDataFile,'\t')
				shownProbeMoments_s=[float(element[0])-startTimePupilRecording_s for element in shownProbeData]
				thisDict['shownProbes']=numpy.array(shownProbeMoments_s)
				
				unreportedProbeMoments_s=[]
				
				for oneShownProbeMoment in shownProbeMoments_s:
					reportsToFollowIt=[element for element in reportedProbeMoments_s if element>oneShownProbeMoment and element<(oneShownProbeMoment+probeResponseWindow_s)]
					if len(reportsToFollowIt)==0:
						unreportedProbeMoments_s=unreportedProbeMoments_s+[oneShownProbeMoment]
						
				thisDict['unreportedProbes']=numpy.array(unreportedProbeMoments_s)
				
			else:
				thisDict['shownProbes']=numpy.array([])			#this is time-aligned to probe
				thisDict['unreportedProbes']=numpy.array([])	#this is time-aligned to probe

			#----------
			#get switch moment, identity, and duration events together to make sure they're all sorted in matching order
			
			switchKinds=['inferredSwitches','physicalSwitches','reportedSwitches']
			switchFileNames=['dirChangeInferredTimesNoReturn','dirChangePhysTimesNoReturns','dirChangeReportTimesNoReturns']
			switchIdentityKinds=['inferredSwitchIdentities','physicalSwitchIdentities','reportedSwitchIdentities']
			switchIdentityFileNames=['dirChangeInferredIdentitiesNoReturn','dirChangePhysIdentitiesNoReturns','dirChangeReportIdentitiesNoReturns']
			switchDurationKinds=['','physicalSwitchDurations','reportedSwitchDurations']
			switchDurationFileNames=['','transDursPhysNoReturns','transDursReportNoReturns']
		
			for oneIndex, oneFileName in enumerate(switchFileNames):
				theTimingFile=[element for element in allRegressorFileNames if 'observer_'+oneObs+'session'+oneSess in element and oneFileName in element][0]
				theTimingData=numpy.loadtxt(myPath+regressorsSubFolder+'/'+theTimingFile)-startTimePupilRecording_s
			
				theIdentityFile=[element for element in allRegressorFileNames if 'observer_'+oneObs+'session'+oneSess in element and switchIdentityFileNames[oneIndex] in element][0]
				theIdentityData=numpy.loadtxt(myPath+regressorsSubFolder+'/'+theIdentityFile)
				
				if not switchDurationFileNames[oneIndex]=='':
					
					theDurationFile=[element for element in allRegressorFileNames if 'observer_'+oneObs+'session'+oneSess in element and switchDurationFileNames[oneIndex] in element][0]
					theDurationData=numpy.loadtxt(myPath+regressorsSubFolder+'/'+theDurationFile)
				else:
					
					theDurationData=[-1 for element in theIdentityData]	#placeholder to allow zipping
			
				combinedData=zip(theTimingData,theIdentityData,theDurationData)
				combinedData.sort()
			
				# sortedTimingData=numpy.loadtxt(myPath+regressorsSubFolder+'/'+theTimingFile)-startTimePupilRecording_s
				# sortedTimingData.sort()
				# 
				# indicesForPicking=[[index for index,value in enumerate(theTimingData) if value==elementInSorted][0] for elementInSorted in sortedTimingData]
				# 
				# theTimingDataSorted=[theTimingData[index] for index in indicesForPicking]
				# theIdentityDataSorted=[theIdentityData[index] for index in indicesForPicking]
			
				thisDict[switchKinds[oneIndex]]=numpy.array([element[0] for element in combinedData])
				thisDict[switchIdentityKinds[oneIndex]]=numpy.array([element[1] for element in combinedData])
				
				if not switchDurationKinds[oneIndex]=='':
					thisDict[switchDurationKinds[oneIndex]]=numpy.array([element[2] for element in combinedData])
				
			#-------------------------
			#if this is a session with active rivalry or any kind of replay, then split the switch events into ones with 0 duration, and of non-zero duration
			
			switchDurationKeys=['physicalSwitchDurations','reportedSwitchDurations']
			switchMomentKeys=['physicalSwitches','reportedSwitches']
			switchMomentSubdividedKeys=[['physicalSwitches0duration','physicalSwitchesNon0duration','physicalSwitchStarts','physicalSwitchEnds'],['reportedSwitches0duration','reportedSwitchesNon0duration','reportedSwitchStarts','reportedSwitchEnds']]

			for oneSwitchKindIndex,oneSwitchDurationKey in enumerate(switchDurationKeys):

				theseDurations=thisDict[oneSwitchDurationKey]

				if not len(theseDurations)==0:

					theseSwitchMoments=thisDict[switchMomentKeys[oneSwitchKindIndex]]
					duration0indices=[durIndex for durIndex,theDur in enumerate(theseDurations) if theDur==0.]
					durationNon0indices=[durIndex for durIndex,theDur in enumerate(theseDurations) if theDur>0.]

					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][0]]=numpy.array([theseSwitchMoments[oneInd] for oneInd in duration0indices])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][1]]=numpy.array([theseSwitchMoments[oneInd] for oneInd in durationNon0indices])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][2]]=numpy.array([theseSwitchMoments[oneInd]-theseDurations[oneInd]/2. for oneInd in durationNon0indices])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][3]]=numpy.array([theseSwitchMoments[oneInd]+theseDurations[oneInd]/2. for oneInd in durationNon0indices])
			
				else:
					
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][0]]=numpy.array([])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][1]]=numpy.array([])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][2]]=numpy.array([])
					thisDict[switchMomentSubdividedKeys[oneSwitchKindIndex][3]]=numpy.array([])
					
			#make the 'paddedPupilSamples' entry in the dict, because you used to do padding
			paddedPupilSamples=numpy.array([element[1] for element in pupilData])
			thisDict['paddedPupilSamples']=paddedPupilSamples
			
			#DIT IS NIEUW OP BASIS VAN CONVERSATIE MET GILLES EN TOMAS IN SEPTEMBER, EN EMAILS OP OCT 1:
			#BEHANDELT PER 60 S TRIAL
			#EERST EXPONENTIAL FITTEN TO INDIVIDUAL TRIAL, EN VAN ACTUAL TIMECOURSES AFTREKKEN
			#DAN LOW-PASS FILTEREN, WEER PER INDIVIDUAL TRIAL
			#DAN Z-SCOREN, WEER PER INDIVIDUAL TRIAL
			#DAN NAAR 0 ZETTEN WHATEVER BUITEN TRIALS VALT
			
			def expofunc(time, gain, tau, offset):
			    return gain*numpy.exp(-tau*time)+offset
			
			minimumTau=1./(10.*float(basicSampleRate))		#don't allow rapid exponentials, because will fit fast pattern in first trial seconds for otherwise flat trials. Whereas this is meant to capture slow drifts
			nyquistFreq=(1./preprocessInfoDict['timePerRowAscFile_s'])/2.
			criticalFreqs=[freq/nyquistFreq for freq in preprocessInfoDict['filterCutoffs_Hz']]	#from http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.butter.html: For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
			
			trialStartIndices=[int(basicSampleRate*oneTrialStart) for oneTrialStart in thisDict['trialStarts']]
			trialEndIndices=[int(basicSampleRate*oneTrialStart)-1 for oneTrialStart in thisDict['trialEnds']]
			
			print('Fitting exponential curves per trial for observer '+oneObs+', session '+oneSess+'.')
			for oneTrialNumber in range(len(trialStartIndices)):
				
				theseIndices=range(trialStartIndices[oneTrialNumber],min(trialEndIndices[oneTrialNumber],len(paddedPupilSamples)))
				
				if [oneObs,oneSess,oneTrialNumber] in [['SG','1',11],['NB','1',5]]:
					
					paddedPupilSamples[theseIndices]=[0. for element in theseIndices]
					print('Setting all samples to 0 for observer '+oneObs+', session '+oneSess+' and trial '+str(oneTrialNumber)+'. That is an exceptional trial that has no data and, it seems, no single event either.')
					
				else:
					
					theseData=paddedPupilSamples[theseIndices]
				
					gainGuesses=[(max(theseData)-min(theseData))/2.,(max(theseData)-min(theseData)),(max(theseData)-min(theseData))*2,-(max(theseData)-min(theseData))/2.,-(max(theseData)-min(theseData)),-(max(theseData)-min(theseData))*2]
					tauGuesses=[1./(11.*basicSampleRate),1./(20.*basicSampleRate),1./(40.*basicSampleRate),1./(80.*basicSampleRate),-1./(11.*basicSampleRate),-1./(20.*basicSampleRate),-1./(40.*basicSampleRate),-1./(80.*basicSampleRate)]
					offsetGuesses=[numpy.average(theseData),max(theseData),min(theseData)]
					
					foundFit=False
					for oneGainGuess in gainGuesses:
						if foundFit:
							break
						for oneTauGuess in tauGuesses:
							if foundFit:
								break
							for oneOffsetGuess in offsetGuesses:
								try:
									print('trying expo fit to trial '+str(oneTrialNumber))
									params,covmat = scipy.optimize.curve_fit(expofunc, range(len(theseData)), theseData, p0=(oneGainGuess, oneTauGuess, oneOffsetGuess), bounds=((-numpy.inf,-minimumTau,-numpy.inf),(numpy.inf,minimumTau,numpy.inf)))	#fit exponential and subtract
									
									foundFit=True
									break
										
								except RuntimeError:
									pass
						
					if foundFit==False:	
						print('no fit found!')
						shell()
					else:
						print('fit found!')
				
					residual=[theseData[index]-expofunc(index,params[0],params[1],params[2]) for index in range(len(theseData))]
			
					fig = pl.figure(figsize = (8,8))
					s = fig.add_subplot(1,1,1)
					pl.plot(range(len(theseData)),theseData, color='k')
					pl.plot(range(len(theseData)),[expofunc(time,params[0],params[1],params[2]) for time in range(len(theseData))], color='k')
					pl.plot(range(len(theseData)),residual, color='g')
				
					b,a=scipy.signal.butter(3, criticalFreqs[1],btype='low')		#filtering low-pass but not high-pass: exponential subtraction takes place of high-pass
					residualClean=scipy.signal.filtfilt(b,a,residual)				
				
					residualClean=(residualClean-numpy.average(residualClean))/numpy.std(residualClean)
				
					#pl.plot(range(len(theseData)),residualClean*150., color='y')
				
					s.set_title('gain: '+str(round(100.*params[0])/100.)+'; 1/tau: '+str(round(100.*1./params[1])/100.)+'; offset: '+str(round(100.*params[2])/100.))
				
					#pl.show()
				
					pl.savefig(myPath+figuresSubFolder+'/observer'+oneObs+'session'+oneSess+'_trial_'+str(oneTrialNumber)+'expofit.pdf')
					pl.close()
				
					numpy.savetxt(myPath+miscStuffSubFolder+'/observer'+oneObs+'session'+oneSess+'_trial_'+str(oneTrialNumber)+'expofitparams.txt',params,fmt='%20.10f')
				
					paddedPupilSamples[theseIndices]=residualClean
				
			betweenTrialIndices=range(0,int(basicSampleRate*thisDict['trialStarts'][0]))
			
			for oneTrialIndex in range(len(thisDict['trialStarts'])):
				
				if oneTrialIndex<(len(thisDict['trialStarts'])-1):
					betweenTrialIndices=betweenTrialIndices+range(int(basicSampleRate*thisDict['trialEnds'][oneTrialIndex])-1,int(basicSampleRate*thisDict['trialStarts'][oneTrialIndex+1]))
			
			betweenTrialIndices=betweenTrialIndices+range(min([int(basicSampleRate*thisDict['trialEnds'][-1])-1,len(paddedPupilSamples)]),len(paddedPupilSamples))
			
			paddedPupilSamples[betweenTrialIndices]=0
			
			#DIT WAS DE OUDE VERSIE, WAAR WE LOW PASS EN HIGH PASS DEDEN IPV LOW PASS EN EXPONENTIAL, EN OP ELK INTERVAL TUSSEN RECALIBRATIES IN PLAATS VAN ELKE TRIAL INDIVIDUALLY
			#
			# recalibrationDataFile=[element for element in allRegressorFileNames if 'observer'+oneObs+'session'+oneSess in element and 'recalibration_periods' in element][0]
			# recalibrationData=filemanipulation.readDelimitedIntoArray(myPath+regressorsSubFolder+'/'+recalibrationDataFile,' ')
			# recalibrationMoments_s=[[float(element[0])-startTimePupilRecording_s,float(element[-1])-startTimePupilRecording_s] for element in recalibrationData]
			#
			# #---------------
			# #filter and then z-score every block of uninterrupted recording (i.e. place cut at each recalibration), and then set all padded stuff that was inserted for recalibration to 0
			# #
			#
			# #the 0 and 1 in postion 2 indicate whether this is a padding interval or a filtering interval
			# filterBlockStartEndIndices=[[0,int(basicSampleRate*recalibrationMoments_s[0][0])-1,0]]
			#
			# for index in range(1,len(recalibrationMoments_s)):
			# 	filterBlockStartEndIndices=filterBlockStartEndIndices+[[int(basicSampleRate*recalibrationMoments_s[index-1][1])+1,int(basicSampleRate*recalibrationMoments_s[index][0])-1,0]]
			# filterBlockStartEndIndices=filterBlockStartEndIndices+[[int(basicSampleRate*recalibrationMoments_s[-1][1])+1,len(paddedPupilSamples),0]]
			#
			# replaceWithZeroStartEndIndices=[[int(basicSampleRate*element[0]),int(basicSampleRate*element[1]),1] for element in recalibrationMoments_s]
			#
			# allIntervalStartEndIndices=filterBlockStartEndIndices+replaceWithZeroStartEndIndices
			# allIntervalStartEndIndices.sort()
			#
			# nyquistFreq=(1./preprocessInfoDict['timePerRowAscFile_s'])/2.
			# criticalFreqs=[freq/nyquistFreq for freq in preprocessInfoDict['filterCutoffs_Hz']]	#from http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.butter.html: For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
			#
			# cleanedPupilSizes=numpy.array([])
			#
			# for oneBlock in allIntervalStartEndIndices:
			#
			# 	if oneBlock[-1]==0:	#this is a filter + z-score block
			#
			# 		b,a=scipy.signal.butter(3, criticalFreqs[1],btype='low')		#filtering low-pass and high-pass consecutively because bandpass doesn't work for some reason
			# 		filteredPupilSizesTemp=scipy.signal.filtfilt(b,a,paddedPupilSamples[oneBlock[0]:oneBlock[1]+1])
			# 		b,a=scipy.signal.butter(3, criticalFreqs[0],btype='high')
			# 		filteredPupilSizes=scipy.signal.filtfilt(b,a,filteredPupilSizesTemp)
			#
			# 		filteredAndZscoredPupilSizes=filteredPupilSizes-numpy.average(filteredPupilSizes)	#z-scoring
			# 		filteredAndZscoredPupilSizes=filteredAndZscoredPupilSizes/numpy.std(filteredPupilSizes)
			#
			# 		cleanedPupilSizes=numpy.concatenate((cleanedPupilSizes,filteredAndZscoredPupilSizes),axis=0)
			#
			# 	elif oneBlock[-1]==1:	#this is a set-to-zero block
			#
			# 		bunchOfZeros=numpy.zeros(oneBlock[1]+1-oneBlock[0])
			# 		cleanedPupilSizes=numpy.concatenate((cleanedPupilSizes,bunchOfZeros),axis=0)
			# 	else:
			# 		raise Exception('OH NOES!')
					
					
			# #To visually inspect that z-scoring and filtering were ok, uncomment the following:
			# timeIntervalsS=200.
			# startTime=0
			# endTime=timeIntervalsS
			#
			# while (startTime+timeIntervalsS)<len(paddedPupilSamples)/float(basicSampleRate):
			#
			# 	endTime=startTime+timeIntervalsS
			#
			# 	theseXvals=[startTime+increment/float(basicSampleRate) for increment in range(int(endTime-startTime)*basicSampleRate)]
			# 	theseYindices=range(int(startTime*basicSampleRate),int(endTime*basicSampleRate))
			#
			# 	theseYvals=paddedPupilSamples[theseYindices]
			#
			# 	#theseYvalsCleaned=cleanedPupilSizes[theseYindices]
			#
			# 	maxY=max(theseYvals)
			# 	minY=min(theseYvals)
			#
			# 	fig = pl.figure(figsize = (8,8))
			# 	s = fig.add_subplot(1,1,1)
			#
			# 	pl.plot(theseXvals,theseYvals, color='k')
			#
			# 	startEvents=[element for element in thisDict['trialStarts'] if element>startTime and element<endTime]
			# 	endEvents=[element for element in thisDict['trialEnds'] if element>startTime and element<endTime]
			#
			# 	if len(startEvents)>0:
			# 		pl.scatter(startEvents,[maxY for index in range(len(startEvents))], color = 'r', marker='o',label='starts',s=10)
			#
			# 	if len(endEvents)>0:
			# 		pl.scatter(endEvents,[minY for index in range(len(endEvents))], color = 'g', marker='o',label='ends',s=10)
			#
			# 	pl.legend()
			# 	pl.show()
			# 	pl.close()
			#
			# 	startTime=endTime

			thisDict['paddedPupilSamplesCleaned']=paddedPupilSamples
			thisDict['paddedPupilSamplesSaccBlinksNotRemoved']=paddedPupilSamples	#STORING UNDER TWO NAMES FOR COMPATIBILTIY WITH DOWNSTREAM STUFF, BUT BOTH ENTRIES ARE THE SAME
			
			#remove all events that are too close to the end or start of a trial. (except trial start/ends themselves as well as signal drop markers)
			switchKinds=['inferredSwitches','physicalSwitches','reportedSwitches']
			switchIdentityKinds=['inferredSwitchIdentities','physicalSwitchIdentities','reportedSwitchIdentities']
			switchDurationKinds=['','physicalSwitchDurations','reportedSwitchDurations']
		
			eventKinds=switchKinds+['blinks','saccades','probeReports','unreportedProbes','shownProbes','physicalSwitches0duration','physicalSwitchesNon0duration','physicalSwitchStarts','physicalSwitchEnds','reportedSwitches0duration','reportedSwitchesNon0duration','reportedSwitchStarts','reportedSwitchEnds']
			
			for myIndex,oneEventKind in enumerate(eventKinds):
				
				theseEvents=thisDict[oneEventKind]
				retainedEvents=[]
				nonRetainedEvents=[]
				
				if 'blinks' in oneEventKind:
					relevantDecoInterval=decoIntervalBlink
				elif 'saccades' in oneEventKind:
					relevantDecoInterval=decoIntervalSacc
				else:
					relevantDecoInterval=decoInterval
					
				if myIndex<len(switchIdentityKinds):	#only the first 3: so only for the switches, where there's the duration and direction arrays to deal with
					thisIdentityKind=switchIdentityKinds[myIndex]
					theseIdentities=thisDict[thisIdentityKind]
					retainedSwitchKinds=[]
					nonRetainedSwitchKinds=[]
					
					if not switchDurationKinds[myIndex]=='':
						thisDurationKind=switchDurationKinds[myIndex]
						theseDurations=thisDict[thisDurationKind]
						retainedSwitchDurations=[]
						nonRetainedSwitchDurations=[]
				
				for oneTrialIndex in range(len(thisDict['trialStarts'])):
				
					retainedIndicesThisTrial=[index for index,element in enumerate(theseEvents) if (element+relevantDecoInterval[0])>thisDict['trialStarts'][oneTrialIndex] and (element+relevantDecoInterval[1])<thisDict['trialEnds'][oneTrialIndex]]	#only retain if far enough from start/end of trial for deco-interval not to run into edges of trial
					# 
					# retainedIndicesThisTrial=[]
					# nonRetainedIndicesThisTrial=[]
					# for oneFirstPassIndex in retainedIndicesThisTrialFirstPass:
					# 
					# 	for oneTimePointPairThatDelimitsMissingData_s in timePointPairsThatDelimitMissingData_s:
					# 
					# 		if (theseEvents[oneFirstPassIndex]+decoInterval[1])<oneTimePointPairThatDelimitsMissingData_s[0]:	#if this oneTimePointPairThatDelimitsMissingData_s's first element happens after the end of this deco window, we can quit because those time point pairs are sorted chronologically
					# 			retainedIndicesThisTrial=retainedIndicesThisTrial+[oneFirstPassIndex]
					# 			break
					# 		elif (theseEvents[oneFirstPassIndex]+decoInterval[0])<oneTimePointPairThatDelimitsMissingData_s[1]:		#if this oneTimePointPairThatDelimitsMissingData_s's first element happens before the end of this deco window, and its second element happens after its start, then we need to delete this event
					# 			nonRetainedIndicesThisTrial=nonRetainedIndicesThisTrial+[oneFirstPassIndex]
					# 			break
					# 		else:		#if this oneTimePointPairThatDelimitsMissingData_s's first element happens before the end of this deco window, but its second element also happens before its start, then let's look at the next oneTimePointPairThatDelimitsMissingData_s
					# 			pass

					retainedEventsThisTrial=[theseEvents[index] for index in retainedIndicesThisTrial]
					retainedEvents=retainedEvents+retainedEventsThisTrial
				
					if myIndex<len(switchIdentityKinds):
						retainedSwitchKindsThisTrial=[theseIdentities[index] for index in retainedIndicesThisTrial]
						retainedSwitchKinds=retainedSwitchKinds+retainedSwitchKindsThisTrial
						
						if not switchDurationKinds[myIndex]=='':
							retainedSwitchDurationsThisTrial=[theseDurations[index] for index in retainedIndicesThisTrial]
							retainedSwitchDurations=retainedSwitchDurations+retainedSwitchDurationsThisTrial
							
					# nonRetainedIndicesThisTrial=[index for index,element in enumerate(theseEvents) if (element-decoInterval[0])<=thisDict['trialStarts'][oneTrialIndex] and (element+decoInterval[1])>=thisDict['trialEnds'][oneTrialIndex]]
					# 
					# nonRetainedEventsThisTrial=[theseEvents[index] for index in nonRetainedIndicesThisTrial]
					# nonRetainedEvents=nonRetainedEvents+nonRetainedEventsThisTrial
					# 
					# if myIndex<len(switchIdentityKinds):
					# 	nonRetainedSwitchKindsThisTrial=[theseIdentities[index] for index in nonRetainedIndicesThisTrial]
					# 	nonRetainedSwitchKinds=nonRetainedSwitchKinds+nonRetainedSwitchKindsThisTrial
					# 
					# 	if not switchDurationKinds[myIndex]=='':
					# 		nonRetainedSwitchDurationsThisTrial=[theseDurations[index] for index in nonRetainedIndicesThisTrial]
					# 		nonRetainedSwitchDurations=nonRetainedSwitchDurations+nonRetainedSwitchDurationsThisTrial
							
				thisDict[oneEventKind]=numpy.array(retainedEvents)
				if myIndex<len(switchIdentityKinds):
					thisDict[thisIdentityKind]=numpy.array(retainedSwitchKinds)
					if not switchDurationKinds[myIndex]=='':
						thisDict[thisDurationKind]=numpy.array(retainedSwitchDurations)
						
				# thisDict[oneEventKind+'_PupilMissing']=numpy.array(nonRetainedEvents)
				# if myIndex<len(switchIdentityKinds):
				# 	thisDict[thisIdentityKind+'_PupilMissing']=numpy.array(nonRetainedSwitchKinds)
				# 	if not switchDurationKinds[myIndex]=='':
				# 		thisDict[thisDurationKind+'_PupilMissing']=numpy.array(nonRetainedSwitchDurations)
				
			# #------------
			# #regress out blink and sacc responses from pupil -- NO DO THIS IN OVERALL GLM!
			#
			# b = FIRDeconvolution.FIRDeconvolution(signal=sp.signal.decimate(paddedPupilSamples, downSampleRate, 1),
			#                          events=[thisDict['blinks'],thisDict['saccades']], event_names=['blinks','saccades'], sample_frequency=newSampleRate,
			#                          deconvolution_frequency=newSampleRate, deconvolution_interval=decoInterval,)
			# b.create_design_matrix()
			#
			# if ridgeForrester:
			# 	b.ridge_regress()
			# else:
			# 	b.regress()
			#
			# b.betas_for_events()
			#
			# for thisIndex,thisName in enumerate(b.covariates.keys()):	#here use the internal .covariates.keys() because there is some sort of shuffling going on internally that determines the order of the regressors
			# 	thisResponse=numpy.array(b.betas_per_event_type[thisIndex]).ravel()
			# 	if thisName=='blinks':
			# 		blinkResponse=thisResponse-numpy.average([thisResponse[0],thisResponse[-1]]) #assuming that the average of the first time point and the last time point is 0, which is reasonable if our deco window is large enough not to cut off a meaningful response
			# 	elif thisName=='saccades':
			# 		saccadeResponse=thisResponse-numpy.average([thisResponse[0],thisResponse[-1]]) #assuming that the average of the first time point and the last time point is 0, which is reasonable if our deco window is large enough not to cut off a meaningful response
			# 	else:
			# 		raise('Something fishy!')
			#
			# # force t=0 through y=0:
			# #time0Index=int(-decoInterval[0]*newSampleRate)
			# #blinkResponse = blinkResponse - blinkResponse[time0Index]
			# #saccadeResponse = saccadeResponse - saccadeResponse[time0Index]
			#
			# eventsForPlot=[blinkResponse,saccadeResponse]
			# eventNamesForPlot=['blinks','saccades']
			#
			# x = numpy.linspace(decoInterval[0],decoInterval[1], len(blinkResponse))
			# f = pl.figure(figsize = (10,4.5))
			#
			# for curveIndex in range(len(eventsForPlot)):
			# 	pl.plot(x, eventsForPlot[curveIndex], color=plotColors[curveIndex], label=eventNamesForPlot[curveIndex])
			#
			# pl.xlabel('Time from event (s)')
			# pl.ylabel('Pupil size')
			# pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
			# pl.legend(loc=2)
			# #sn.despine(offset=10)
			#
			# if ridgeForrester:
			# 	pl.savefig(myPath+figuresSubFolder+'/observer'+oneObs+'session'+oneSess+'_saccade_and_blink_response_'+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_ridge.pdf')
			# else:
			# 	pl.savefig(myPath+figuresSubFolder+'/observer'+oneObs+'session'+oneSess+'_saccade_and_blink_response_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'.pdf')
			#
			# pl.close()
			#
			# thisDict['blinkResponse']=blinkResponse
			# thisDict['saccadeResponse']=saccadeResponse
			
			#--------------------------
			#if this is a session with active rivalry or active replay, then determine the temporal relation between reported/physical switches and the inferred ones, to shift event times later
			#for this particular analysis don't use the 'pupil_OK' version because it doesn't involve pupils
			
			rawBehavFile=[element for element in allBehavioralFileNames if 'observer_'+oneObs+'_session_'+oneSess in element][0]
			sessionKindIndicator=re.findall('.*_rivalryReplay_([0-9]*)_reportSwitchesProbes_([0-9]*)_time.*',rawBehavFile)[0]
		
			if sessionKindIndicator==('0','0'):
				thisDict['sessionKind']='Active rivalry'
			elif sessionKindIndicator==('0','1'):
				thisDict['sessionKind']='Passive rivalry'
			elif sessionKindIndicator==('1','0'):
				thisDict['sessionKind']='Active replay'
			elif sessionKindIndicator==('1','1'):
				thisDict['sessionKind']='Passive replay'
				
			if thisDict['sessionKind'] in ['Active rivalry','Active replay']:
				
				if thisDict['sessionKind']=='Active rivalry':
					decoInterval_switches=decoInterval_switches_riv_inf_rep
				else:
					decoInterval_switches=decoInterval_switches_repl_inf_rep
				
				#the following was altered Nov 14 2018
				#----------
				reportedSwitchResponsesPerPercept=[]
				f = pl.figure(figsize = (10,4.5))
				colors=['r','g']
				for identityIndex,identity in enumerate([-1,1]):
				
					# OLD APPROACH BUT SP.SIGNAL.DECIMATE BEHAVED WEIRDLY RE Y-AXIS SCALING
					# inferred_timecourse=numpy.zeros(len(thisDict['paddedPupilSamples']))
					# inferredSwitchIndices=[int(element*float(basicSampleRate)) for index, element in enumerate(thisDict['inferredSwitches']) if thisDict['inferredSwitchIdentities'][index]==identity]
					#
					# inferredSwitchIndicesNew=[index for index in inferredSwitchIndices if index<len(inferred_timecourse)]
					#
					# numSwitchesDropped=len(inferredSwitchIndices)-len(inferredSwitchIndicesNew)
					# if numSwitchesDropped>0:
					# 	print 'Watch out! Dropping '+str(numSwitchesDropped)+' inferred switches for '+oneObs+', session '+str(oneSess)+' when determining delay between inferred switches and reported ones.'
					# 	print 'I suspect this may be due to missing eye data at the end of a session so it shouldn\'t happen more than incidentally'
					# 	inferredSwitchIndices=inferredSwitchIndicesNew
					#
					# inferred_timecourse[inferredSwitchIndices]=1	#turn it into a time series with 0s and 1s to perform deconvolution
					#
					# theseEvents=numpy.array([element for index,element in enumerate(thisDict['reportedSwitches']) if thisDict['reportedSwitchIdentities'][index]==identity])
					#
					# b = FIRDeconvolution.FIRDeconvolution(signal=sp.signal.decimate(inferred_timecourse, downSampleRateBehavioralSwitchToSwitch, 1),
					#                          events=[theseEvents], event_names=['reported'], sample_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch,
					#                          deconvolution_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, deconvolution_interval=decoInterval_switches,)
					#
					
					inferred_timecourse=numpy.zeros(int(len(thisDict['paddedPupilSamples'])/downSampleRateBehavioralSwitchToSwitch))
					inferredSwitchIndices=[int(element*float(basicSampleRate)/float(downSampleRateBehavioralSwitchToSwitch)) for index, element in enumerate(thisDict['inferredSwitches']) if thisDict['inferredSwitchIdentities'][index]==identity]
					
					inferredSwitchIndicesNew=[index for index in inferredSwitchIndices if index<len(inferred_timecourse)]
					
					numSwitchesDropped=len(inferredSwitchIndices)-len(inferredSwitchIndicesNew)
					if numSwitchesDropped>0:
						print 'Watch out! Dropping '+str(numSwitchesDropped)+' inferred switches for '+oneObs+', session '+str(oneSess)+' when determining delay between inferred switches and reported ones.'
						print 'I suspect this may be due to missing eye data at the end of a session so it shouldn\'t happen more than incidentally'
						inferredSwitchIndices=inferredSwitchIndicesNew
					
					inferred_timecourse[inferredSwitchIndices]=1	#turn it into a time series with 0s and 1s to perform deconvolution
					
					theseEvents=numpy.array([element for index,element in enumerate(thisDict['reportedSwitches']) if thisDict['reportedSwitchIdentities'][index]==identity])
				
					b = FIRDeconvolution.FIRDeconvolution(signal=inferred_timecourse, 
					                         events=[theseEvents], event_names=['reported'], sample_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, 
					                         deconvolution_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, deconvolution_interval=decoInterval_switches,)

					b.create_design_matrix(intercept=False)

					if ridgeForrester:
						b.ridge_regress()
					else:
						b.regress()
					b.betas_for_events()
			
					reportedSwitchResponse=numpy.array(b.betas_per_event_type[0]).ravel()
					reportedSwitchResponsesPerPercept=reportedSwitchResponsesPerPercept+[reportedSwitchResponse]
				
					x = numpy.linspace(decoInterval_switches[0],decoInterval_switches[1], len(reportedSwitchResponse))
					pl.plot(x, reportedSwitchResponse, color=colors[identityIndex], linewidth=.5, label='reported '+str(identity))
				
				reportedSwitchResponse=numpy.average(reportedSwitchResponsesPerPercept,axis=0)	
			
				pl.plot(x, reportedSwitchResponse, linewidth=1., label='average ',color='k')
				peakTime=[thisX for index,thisX in enumerate(x) if reportedSwitchResponse[index]==max(reportedSwitchResponse)][0]
				pl.plot([peakTime,peakTime],[min(reportedSwitchResponse),max(reportedSwitchResponse)],color='k')

				pl.xlabel('Time from event (s)')
				pl.ylabel('Prop density inferred')
				pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
				pl.legend(loc=2)
				#sn.despine(offset=10)
				#----------
			
				pl.savefig(myPath+figuresSubFolder+'/'+str(oneObs)+'_session_'+str(oneSess)+'_switch_related_decos_'+thisDict['sessionKind']+'_reported.pdf')
				pl.close()
				
				thisDict['reportedInferredDecoPlot']=[x, reportedSwitchResponse]
				
				if thisDict['sessionKind']=='Active rivalry':
					thisDict['rivalryReportedToInferredShift']=peakTime	#how much to shift reported rivalry switch times to align them as closely as possible with inferred
					thisDict['rivalryInferredToReportedShift']=-peakTime #how much to shift inferred rivalry switch times to align them as closely as possible with where the key press report would have been
				else:
					thisDict['replayReportedToInferredShift']=peakTime	#how much to shift reported replay switch times to align them as closely as possible with inferred
					thisDict['replayInferredToReportedShift']=-peakTime	#how much to shift reported replay switch times to align them as closely as possible with where the key press report would have been

				if thisDict['sessionKind']=='Active replay':
				
					#the following was altered Nov 14
					#----------
					
					decoInterval_switches=decoInterval_switches_repl_inf_phys
				
					physicalSwitchResponsesPerPercept=[]		#this is inferred density surrounding physical
					f = pl.figure(figsize = (10,4.5))
				
					for identityIndex,identity in enumerate([-1,1]):
						
						inferred_timecourse=numpy.zeros(int(len(thisDict['paddedPupilSamples'])/downSampleRateBehavioralSwitchToSwitch))
						inferredSwitchIndices=[int(element*float(basicSampleRate)/float(downSampleRateBehavioralSwitchToSwitch)) for index, element in enumerate(thisDict['inferredSwitches']) if thisDict['inferredSwitchIdentities'][index]==identity]
						
						inferredSwitchIndicesNew=[index for index in inferredSwitchIndices if index<len(inferred_timecourse)]
					
						numSwitchesDropped=len(inferredSwitchIndices)-len(inferredSwitchIndicesNew)
						if numSwitchesDropped>0:
							print 'Watch out! Dropping '+str(numSwitchesDropped)+' inferred switches for '+oneObs+', session '+str(oneSess)+' when determining delay between inferred switches and physical ones.'
							print 'I suspect this may be due to missing eye data at the end of a session so it shouldn\'t happen more than incidentally'
							inferredSwitchIndices=inferredSwitchIndicesNew
					
						inferred_timecourse[inferredSwitchIndices]=1	#turn it into a time series with 0s and 1s to perform deconvolution
					
						theseEvents=numpy.array([element for index,element in enumerate(thisDict['physicalSwitches']) if thisDict['physicalSwitchIdentities'][index]==identity])
						
						# startTime=0
						# endTime=10
						# maxtime=theseEvents[-1]
						# while endTime<maxtime:
						# 	theseInferred=[element/float(basicSampleRate)+startTimePupilRecording_s for element in inferredSwitchIndices if (element/basicSampleRate+startTimePupilRecording_s)>startTime and (element/basicSampleRate+startTimePupilRecording_s)<endTime]
						# 	thesePhys=[element+startTimePupilRecording_s for element in theseEvents if (element+startTimePupilRecording_s)>startTime and (element+startTimePupilRecording_s)<endTime]
						# 	
						# 	f = pl.figure(figsize = (10,4.5))
						# 	s = f.add_subplot(1,1,1)
						# 	s.scatter([theseInferred],[.5 for aap in theseInferred],color='r')
						# 	s.scatter([thesePhys],[1. for aap in thesePhys],color='b')
						# 	
						# 	s.set_xlim(xmin =startTime, xmax = endTime)
						# 	s.set_ylim(ymin =0.25, ymax = 1.25)
						# 
						# 	pl.savefig('/Users/janbrascamp/Desktop/_'+str(startTime)+'.pdf')
						# 	pl.close()
						# 	
						# 	startTime=startTime+10
						# 	endTime=endTime+10
						
						b = FIRDeconvolution.FIRDeconvolution(signal=inferred_timecourse, 
						                         events=[theseEvents], event_names=['physical'], sample_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, 
						                         deconvolution_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, deconvolution_interval=decoInterval_switches,)
						b.create_design_matrix(intercept=False)

						if ridgeForrester:
							b.ridge_regress()
						else:
							b.regress()
						b.betas_for_events()
					
						physicalSwitchResponse=numpy.array(b.betas_per_event_type[0]).ravel()
						physicalSwitchResponsesPerPercept=physicalSwitchResponsesPerPercept+[physicalSwitchResponse]
					
						x = numpy.linspace(decoInterval_switches[0],decoInterval_switches[1], len(physicalSwitchResponse))
					
						pl.plot(x, physicalSwitchResponse, color=colors[identityIndex], linewidth=.5, label='physical '+str(identity))
					
					physicalSwitchResponse=numpy.average(physicalSwitchResponsesPerPercept,axis=0)	
					pl.plot(x, physicalSwitchResponse, linewidth=1., label='average ',color='k')
					peakTime=[thisX for index,thisX in enumerate(x) if physicalSwitchResponse[index]==max(physicalSwitchResponse)][0]
					pl.plot([peakTime,peakTime],[min(physicalSwitchResponse),max(physicalSwitchResponse)],color='k')

					thisDict['replayPhysicalToInferredShift']=peakTime	#how much to shift physical replay switch times to align them as closely as possible with inferred
				
					#----------------
				
					pl.xlabel('Time from event (s)')
					pl.ylabel('Prop density inferred')
					pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
					pl.legend(loc=2)
					#sn.despine(offset=10)

					pl.savefig(myPath+figuresSubFolder+'/'+str(oneObs)+'_session_'+str(oneSess)+'_switch_related_decos_'+thisDict['sessionKind']+'_physical.pdf')
					pl.close()
					
					thisDict['physicalInferredDecoPlot']=[x, physicalSwitchResponse]
					
					#--------repeat the same but for the reported, rather than inferred, percepts relative to physical
					
					decoInterval_switches=decoInterval_switches_repl_rep_phys
					
					physicalRepSwitchResponsesPerPercept=[]		#this is reported density surrounding physical
					f = pl.figure(figsize = (10,4.5))
				
					for identityIndex,identity in enumerate([-1,1]):
						
						reported_timecourse=numpy.zeros(int(len(thisDict['paddedPupilSamples'])/downSampleRateBehavioralSwitchToSwitch))
						reportedSwitchIndices=[int(element*float(basicSampleRate)/float(downSampleRateBehavioralSwitchToSwitch)) for index, element in enumerate(thisDict['reportedSwitches']) if thisDict['reportedSwitchIdentities'][index]==identity]
					
						reportedSwitchIndicesNew=[index for index in reportedSwitchIndices if index<len(reported_timecourse)]
					
						numSwitchesDropped=len(reportedSwitchIndices)-len(reportedSwitchIndicesNew)
						if numSwitchesDropped>0:
							print 'Watch out! Dropping '+str(numSwitchesDropped)+' reported switches for '+oneObs+', session '+str(oneSess)+' when determining delay between reported switches and physical ones.'
							print 'I suspect this may be due to missing eye data at the end of a session so it shouldn\'t happen more than incidentally'
							reportedwitchIndices=reportedSwitchIndicesNew
					
						reported_timecourse[reportedSwitchIndices]=1	#turn it into a time series with 0s and 1s to perform deconvolution
					
						theseEvents=numpy.array([element for index,element in enumerate(thisDict['physicalSwitches']) if thisDict['physicalSwitchIdentities'][index]==identity])
				
						b = FIRDeconvolution.FIRDeconvolution(signal=reported_timecourse, 
						                         events=[theseEvents], event_names=['physical'], sample_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, 
						                         deconvolution_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, deconvolution_interval=decoInterval_switches,)
						b.create_design_matrix(intercept=False)

						if ridgeForrester:
							b.ridge_regress()
						else:
							b.regress()
						b.betas_for_events()
					
						physicalRepSwitchResponse=numpy.array(b.betas_per_event_type[0]).ravel()
						physicalRepSwitchResponsesPerPercept=physicalRepSwitchResponsesPerPercept+[physicalRepSwitchResponse]
					
						x = numpy.linspace(decoInterval_switches[0],decoInterval_switches[1], len(physicalRepSwitchResponse))
					
						pl.plot(x, physicalRepSwitchResponse, color=colors[identityIndex], linewidth=.5, label='physical '+str(identity))
					
					physicalRepSwitchResponse=numpy.average(physicalRepSwitchResponsesPerPercept,axis=0)	
					pl.plot(x, physicalRepSwitchResponse, linewidth=1., label='average ',color='k')
					peakTime=[thisX for index,thisX in enumerate(x) if physicalRepSwitchResponse[index]==max(physicalRepSwitchResponse)][0]
					pl.plot([peakTime,peakTime],[min(physicalRepSwitchResponse),max(physicalRepSwitchResponse)],color='k')

					thisDict['replayPhysicalToReportedShift']=peakTime	#how much to shift physical replay switch times to align them as closely as possible with inferred
				
					#----------------
				
					pl.xlabel('Time from event (s)')
					pl.ylabel('Prop density reported')
					pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
					pl.legend(loc=2)
					#sn.despine(offset=10)

					pl.savefig(myPath+figuresSubFolder+'/'+str(oneObs)+'_session_'+str(oneSess)+'_switch_related_decos_'+thisDict['sessionKind']+'_physicalRep.pdf')
					pl.close()
					
					thisDict['physicalReportedDecoPlot']=[x, physicalRepSwitchResponse]
					
					#------------------
					
			elif thisDict['sessionKind']=='Passive replay':
				
				colors=['r','g']
				decoInterval_switches=decoInterval_switches_repl_inf_phys
			
				physicalSwitchResponsesPerPercept=[]		#this is inferred density surrounding physical
				f = pl.figure(figsize = (10,4.5))
			
				for identityIndex,identity in enumerate([-1,1]):
					
					inferred_timecourse=numpy.zeros(int(len(thisDict['paddedPupilSamples'])/downSampleRateBehavioralSwitchToSwitch))
					inferredSwitchIndices=[int(element*float(basicSampleRate)/float(downSampleRateBehavioralSwitchToSwitch)) for index, element in enumerate(thisDict['inferredSwitches']) if thisDict['inferredSwitchIdentities'][index]==identity]
					
					inferredSwitchIndicesNew=[index for index in inferredSwitchIndices if index<len(inferred_timecourse)]
				
					numSwitchesDropped=len(inferredSwitchIndices)-len(inferredSwitchIndicesNew)
					if numSwitchesDropped>0:
						print 'Watch out! Dropping '+str(numSwitchesDropped)+' inferred switches for '+oneObs+', session '+str(oneSess)+' when determining delay between inferred switches and physical ones.'
						print 'I suspect this may be due to missing eye data at the end of a session so it shouldn\'t happen more than incidentally'
						inferredSwitchIndices=inferredSwitchIndicesNew
				
					inferred_timecourse[inferredSwitchIndices]=1	#turn it into a time series with 0s and 1s to perform deconvolution
				
					theseEvents=numpy.array([element for index,element in enumerate(thisDict['physicalSwitches']) if thisDict['physicalSwitchIdentities'][index]==identity])
					
					b = FIRDeconvolution.FIRDeconvolution(signal=inferred_timecourse, 
					                         events=[theseEvents], event_names=['physical'], sample_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, 
					                         deconvolution_frequency=basicSampleRate/downSampleRateBehavioralSwitchToSwitch, deconvolution_interval=decoInterval_switches,)
					b.create_design_matrix(intercept=False)

					if ridgeForrester:
						b.ridge_regress()
					else:
						b.regress()
					b.betas_for_events()
				
					physicalSwitchResponse=numpy.array(b.betas_per_event_type[0]).ravel()
					physicalSwitchResponsesPerPercept=physicalSwitchResponsesPerPercept+[physicalSwitchResponse]
				
					x = numpy.linspace(decoInterval_switches[0],decoInterval_switches[1], len(physicalSwitchResponse))
				
					pl.plot(x, physicalSwitchResponse, color=colors[identityIndex], linewidth=.5, label='physical '+str(identity))
				
				physicalSwitchResponse=numpy.average(physicalSwitchResponsesPerPercept,axis=0)	
				pl.plot(x, physicalSwitchResponse, linewidth=1., label='average ',color='k')
				peakTime=[thisX for index,thisX in enumerate(x) if physicalSwitchResponse[index]==max(physicalSwitchResponse)][0]
				pl.plot([peakTime,peakTime],[min(physicalSwitchResponse),max(physicalSwitchResponse)],color='k')

				thisDict['passiveReplayPhysicalToInferredShift']=peakTime	#how much to shift physical replay switch times to align them as closely as possible with inferred
			
				#----------------
			
				pl.xlabel('Time from event (s)')
				pl.ylabel('Prop density inferred')
				pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
				pl.legend(loc=2)
				#sn.despine(offset=10)

				pl.savefig(myPath+figuresSubFolder+'/'+str(oneObs)+'_session_'+str(oneSess)+'_switch_related_decos_'+thisDict['sessionKind']+'_physical.pdf')
				pl.close()
				
				thisDict['passivePhysicalInferredDecoPlot']=[x, physicalSwitchResponse]
					
			#------------------------------
				
			allInfoDictListThisObs=allInfoDictListThisObs+[thisDict]

		#NO: BLINKS AND SACCADES ARE TAKEN CARE OF IN OVERALL GLM
		# #----------------------
		# #determine average blink / saccade response and regress it out of the time course
		# averageBlinkResponseThisObs=numpy.average(numpy.array([oneDict['blinkResponse'] for oneDict in allInfoDictListThisObs]),0)
		# averageSaccadeResponseThisObs=numpy.average(numpy.array([oneDict['saccadeResponse'] for oneDict in allInfoDictListThisObs]),0)
		#
		# eventsForPlot=[averageBlinkResponseThisObs,averageSaccadeResponseThisObs]
		# eventNamesForPlot=['blinks','saccades']
		#
		# x = numpy.linspace(decoInterval[0],decoInterval[1], len(averageBlinkResponseThisObs))
		# f = pl.figure(figsize = (10,4.5))
		#
		# for curveIndex in range(len(eventsForPlot)):
		# 	pl.plot(x, eventsForPlot[curveIndex], color=plotColors[curveIndex], label=eventNamesForPlot[curveIndex])
		#
		# pl.xlabel('Time from event (s)')
		# pl.ylabel('Pupil size')
		# pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
		# pl.legend(loc=2)
		# #sn.despine(offset=10)
		#
		# if ridgeForrester:
		# 	pl.savefig(myPath+figuresSubFolder+'/observer'+oneObs+'_averagedAcrossSessions_saccade_and_blink_response_'+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_ridge.pdf')
		# else:
		# 	pl.savefig(myPath+figuresSubFolder+'/observer'+oneObs+'_averagedAcrossSessions_saccade_and_blink_response_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'.pdf')
		#
		# pl.close()
		#
		# numpy.save(myPath+miscStuffSubFolder+'/'+oneObs+'_averageBlinkResponse_'+str(decoInterval[0])+'-'+str(decoInterval[1]),[x,averageBlinkResponseThisObs])
		# numpy.save(myPath+miscStuffSubFolder+'/'+oneObs+'_averageSaccResponse_'+str(decoInterval[0])+'-'+str(decoInterval[1]),[x,averageSaccadeResponseThisObs])
		#
		# for infoDictIndex,oneInfoDictThisObs in enumerate(allInfoDictListThisObs):
		#
		# 	#bug discovered on Sept 30 2020: this undoes filtering and z-scoring
		# 	#paddedPupilSamples=oneInfoDictThisObs['paddedPupilSamples']
		#
		# 	#x=numpy.linspace(decoInterval[0],decoInterval[1], (decoInterval[1]-decoInterval[0])*basicSampleRate)
		# 	#blinkKernel=averageBlinkResponseThisObs
		# 	#saccadeKernel=averageSaccadeResponseThisObs
		#
		# 	blinkTimes=oneInfoDictThisObs['blinks']
		# 	saccadeTimes=oneInfoDictThisObs['saccades']
		#
		# 	blinkKernel=averageBlinkResponseThisObs.repeat(downSampleRate,axis=0)	#This shifts the kernel by downsample_rate. So in the next line that's added again.
		# 	saccadeKernel=averageSaccadeResponseThisObs.repeat(downSampleRate,axis=0)	#This shifts the kernel by downsample_rate. So in the next line that's added again.
		#
		# 	blinkSampleIndices=[int((element+decoInterval[0])*basicSampleRate-int(downSampleRate)) for element in blinkTimes]
		# 	saccadeSampleIndices=[int((element+decoInterval[0])*basicSampleRate-int(downSampleRate)) for element in saccadeTimes]
		#
		# 	blinkRegressor=numpy.zeros(len(paddedPupilSamples))
		# 	blinkRegressor[blinkSampleIndices]=1
		# 	blinkRegConv=sp.signal.fftconvolve(blinkRegressor, blinkKernel, 'full')[:-(len(blinkKernel)-1)]
		# 	blinkRegConv=blinkRegConv-numpy.average(blinkRegConv)
		#
		# 	saccadeRegressor=numpy.zeros(len(paddedPupilSamples))
		# 	saccadeRegressor[saccadeSampleIndices]=1
		# 	saccRegConv=sp.signal.fftconvolve(saccadeRegressor, saccadeKernel, 'full')[:-(len(saccadeKernel)-1)]
		# 	saccRegConv=saccRegConv-numpy.average(saccRegConv)
		#
		# 	#-------
		# 	# print 'watskebeurt?'
		# 	# shell()
		# 	#
		# 	# howmuch=40000
		# 	# f = pl.figure(figsize = (10,4.5))
		# 	# pl.plot(range(howmuch), [blinkRegConv[index] for index in range(howmuch)])
		# 	# pl.savefig(myPath+figuresSubFolder+'/__toinevanpeeperstraten.pdf')
		# 	# pl.close()
		# 	#--------
		#
		# 	regs=[]
		# 	if blinkSampleIndices:
		# 		regs=regs+[blinkRegConv]
		#
		# 	if saccadeSampleIndices:
		# 		regs=regs+[saccRegConv]
		#
		# 	# GLM:	I don't understand why we need to adjust scaling again via fitted beta weight but apparently we do.
		# 	designMatrix=numpy.matrix(numpy.vstack([reg for reg in regs])).T
		# 	#betas=numpy.array(((designMatrix.T * designMatrix).I * designMatrix.T) * numpy.matrix(paddedPupilSamples).T).ravel()
		# 	betas=numpy.array([1.,1.])
		#
		# 	#explained = numpy.sum(numpy.vstack([1.*regs[i] for i in range(len(regs))]), axis=0)
		# 	explained=numpy.sum(numpy.vstack([betas[i]*regs[i] for i in range(len(betas))]), axis=0)
		#
		# 	# clean pupil:
		# 	paddedPupilSamplesCleaned = paddedPupilSamples - explained
		# 	paddedPupilSamplesSaccBlinksNotRemoved = paddedPupilSamples		#keep this too, to regress saccades and blinks out in larger GLM that also includes other regressors.
		#
		# 	#paddedPupilSamplesCleaned = paddedPupilSamplesCleaned-numpy.average(paddedPupilSamplesCleaned)		#good idea?
		#
		# 	#-------
		# 	# print 'watskebeurt 2?'
		# 	# shell()
		# 	#
		# 	# howmuch=40000
		# 	# f = pl.figure(figsize = (10,4.5))
		# 	# pl.plot(range(howmuch), [paddedPupilSamplesCleaned[index] for index in range(howmuch)])
		# 	# pl.savefig(myPath+figuresSubFolder+'/__harmensiezen.pdf')
		# 	# pl.close()
		# 	#-------
		#
		# 	#zero-mean everything again after this cleanup: set between-trial periods to 0 and set the rest so that their
		# 	#average is 0
		# 	withinTrialIndices=[]
		# 	betweenTrialIndices=range(0,int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialStarts'][0]))
		#
		# 	for oneTrialIndex in range(len(allInfoDictListThisObs[infoDictIndex]['trialStarts'])):
		#
		# 		withinTrialIndices=withinTrialIndices+range(int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialStarts'][oneTrialIndex]),min(int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialEnds'][oneTrialIndex]),len(paddedPupilSamplesCleaned)))
		#
		# 		if oneTrialIndex<(len(allInfoDictListThisObs[infoDictIndex]['trialStarts'])-1):
		# 			betweenTrialIndices=betweenTrialIndices+range(int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialEnds'][oneTrialIndex]),int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialStarts'][oneTrialIndex+1]))
		#
		# 	betweenTrialIndices=betweenTrialIndices+range(min([int(basicSampleRate*allInfoDictListThisObs[infoDictIndex]['trialEnds'][-1]),len(paddedPupilSamplesCleaned)]),len(paddedPupilSamplesCleaned))
		#
		# 	paddedPupilSamplesCleaned=paddedPupilSamplesCleaned-numpy.average(paddedPupilSamplesCleaned[withinTrialIndices])
		# 	paddedPupilSamplesCleaned[betweenTrialIndices]=0
		#
		# 	paddedPupilSamplesSaccBlinksNotRemoved=paddedPupilSamplesSaccBlinksNotRemoved-numpy.average(paddedPupilSamplesSaccBlinksNotRemoved[withinTrialIndices])
		# 	paddedPupilSamplesSaccBlinksNotRemoved[betweenTrialIndices]=0
		#
		# 	allInfoDictListThisObs[infoDictIndex]['paddedPupilSamplesCleaned']=paddedPupilSamplesCleaned	#this has received all the preprocessing that ['paddedPupilSamples'] has, but in addition blinks and saccaded have been regressed out.
		# 	allInfoDictListThisObs[infoDictIndex]['paddedPupilSamplesSaccBlinksNotRemoved']=paddedPupilSamplesSaccBlinksNotRemoved		#this is identical to ['paddedPupilSamples']
		# 	#-----------------------
		#
		# 	eventIndices=[blinkSampleIndices,saccadeSampleIndices]
		# 	regConvs=[blinkRegConv*betas[0],saccRegConv*betas[1]]
		# 	eventNamesForPlot=['blink','saccade']
		# 	kernels=[blinkKernel,saccadeKernel]
		#
		# 	for eventTypeIndex,eventIndicesOneEvent in enumerate(eventIndices):	#to visually inspect that cleaning up has been done well, show event-related average
		#
		# 		originalSignals=numpy.array([paddedPupilSamples[oneEventIndex:int(oneEventIndex+(decoInterval[1]-decoInterval[0])*basicSampleRate)] for oneEventIndex in eventIndicesOneEvent if oneEventIndex>0 and oneEventIndex<(len(paddedPupilSamples)-(decoInterval[1]-decoInterval[0])*basicSampleRate-1)])
		# 		originalSignals=numpy.average(originalSignals,axis=0)
		#
		# 		cleanedSignals=numpy.array([paddedPupilSamplesCleaned[oneEventIndex:int(oneEventIndex+(decoInterval[1]-decoInterval[0])*basicSampleRate)] for oneEventIndex in eventIndicesOneEvent if oneEventIndex>0 and oneEventIndex<(len(paddedPupilSamplesCleaned)-(decoInterval[1]-decoInterval[0])*basicSampleRate-1)])
		# 		cleanedSignals=numpy.average(cleanedSignals,axis=0)
		#
		# 		regConvsThisEvent=numpy.array([regConvs[eventTypeIndex][oneEventIndex:int(oneEventIndex+(decoInterval[1]-decoInterval[0])*basicSampleRate)] for oneEventIndex in eventIndicesOneEvent if oneEventIndex>0 and oneEventIndex<(len(regConvs[eventTypeIndex])-(decoInterval[1]-decoInterval[0])*basicSampleRate-1)])
		# 		regConvsThisEvent=numpy.average(regConvsThisEvent,axis=0)
		#
		# 		f = pl.figure(figsize = (10,4.5))
		#
		# 		x = range(len(originalSignals))
		# 		pl.plot(x, originalSignals, label='original', linewidth=.5)
		# 		pl.plot(x, cleanedSignals, label='cleaned', linewidth=.5)
		# 		pl.plot(x, regConvsThisEvent, label='avg regressor, beta '+str(numpy.round(1000*betas[eventTypeIndex])/1000), linewidth=.5)
		# 		pl.plot(x, kernels[eventTypeIndex], label='kernel', linewidth=.5)
		#
		# 		pl.xlabel('Time (ms)')
		# 		pl.ylabel('Pupil size')
		# 		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
		# 		pl.legend()
		# 		#sn.despine(offset=10)
		#
		# 		if ridgeForrester:
		# 			pl.savefig(myPath+figuresSubFolder+'/'+oneObs+'_eventRelatedAverageAfterRegression_'+eventNamesForPlot[eventTypeIndex]+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_'+str(numpy.round(1000*betas[eventTypeIndex])/1000)+'_ridge.pdf')
		# 		else:
		# 			pl.savefig(myPath+figuresSubFolder+'/'+oneObs+'_eventRelatedAverageAfterRegression_'+eventNamesForPlot[eventTypeIndex]+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_'+str(numpy.round(1000*betas[eventTypeIndex])/1000)+'.pdf')
		#
		# 		pl.close()
		
		with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', 'w') as f:
		    pickle.dump(allInfoDictListThisObs, f)
			
	allInfoDictList=allInfoDictList+allInfoDictListThisObs	#not nesting the observers; just going to pick out the observers etc later. Good idea?

# ALL GLM STUFF BELOW IS NOW DONE USING THE GILLES (FOURIER) APPROACH
# #Now that it's tidy, run interesting GLMS and compare.
#
# downSampleRate=100
# newSampleRate=basicSampleRate/downSampleRate
# #decoInterval=[-3.5,5.]		let's not redefine it: for _PupilOK versions we removed events that were too close to rubbish based on decoInterval that was current there
# baselineInterval=[-3.5,-3.]
# ridgeForrester=False
#
# plotDictList=[]
#
# # #-----------
# # #define all the GLMS
# #
# # #first aligned to the inferred switch moment
# # plotTitle='Active_rivalry_reported_vs_inferred_aligned_w_OKN'
# # sessionKinds=['Active rivalry','Active rivalry']
# # regressors=[['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0]]
# # timeShifts=[[['Active rivalry','rivalryReportedToInferredShift'],0],[0,0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# # alignmentInfo=[['OKN'],['OKN']]	#information of regressors that trail what will be plotted, do not have to be filled out
# # calculate_var_explained=[[1,0],[2,0]]		#1: the one that is to be explained; 2: the one that does the explaining
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_replay_reported_vs_inferred_vs_physical_aligned_w_OKN'
# # sessionKinds=['Active replay','Active replay','Active replay']
# # regressors=[['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['physicalSwitches_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0]]
# # timeShifts=[[['Active replay','replayReportedToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0]]
# # alignmentInfo=[['OKN'],['OKN'],['OKN']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_vs_passive_rivalry_inferred_aligned_w_OKN'
# # sessionKinds=['Active rivalry','Passive rivalry']
# # regressors=[['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0,1,2]]
# # timeShifts=[[0,0],[0,0,0,0]]
# # alignmentInfo=[['OKN','n/a'],['OKN','n/a','n/a']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_vs_passive_replay_physical_vs_inferred_aligned_w_OKN'
# # sessionKinds=['Active replay','Active replay','Passive replay','Passive replay']
# # regressors=[['physicalSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['physicalSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0,1,2],[0]]
# # timeShifts=[[['Active replay','replayPhysicalToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0,0,0],[0,0,0,0]]
# # alignmentInfo=[['OKN'],['OKN','n/a'],['OKN','n/a','n/a'],['OKN','n/a','n/a']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_and_passive_rivalry_vs_replay_inferred_aligned_w_OKN'
# # sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
# # regressors=[['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0],[0]]
# # timeShifts=[[0,0],[0,0],[0,0,0,0],[0,0,0,0]]
# # alignmentInfo=[['OKN'],['OKN'],['OKN'],['OKN']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # #----and key press aligned
# # plotTitle='Active_rivalry_reported_vs_inferred_aligned_w_key'
# # sessionKinds=['Active rivalry','Active rivalry']
# # regressors=[['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0]]
# # timeShifts=[[0,0],[['Active rivalry','rivalryInferredToReportedShift'],0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# # alignmentInfo=[['keys'],['keys']]
# # calculate_var_explained=[[1,0],[2,0]]		#1: the one that is to be explained; 2: the one that does the explaining
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_vs_passive_rivalry_inferred_aligned_w_key'
# # sessionKinds=['Active rivalry','Passive rivalry']
# # regressors=[['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0,1,2]]
# # timeShifts=[[['Active rivalry','rivalryInferredToReportedShift'],0],[['Active rivalry','rivalryInferredToReportedShift'],0,0,0]]
# # alignmentInfo=[['keys'],['keys','n/a','n/a']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_and_passive_rivalry_vs_replay_inferred_aligned_w_key'
# # sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
# # regressors=[['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts'],['inferredSwitches_PupilOK','probeReports_PupilOK','unreportedProbes_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0],[0]]
# # timeShifts=[[['Active rivalry','rivalryInferredToReportedShift'],0],[['Active replay','replayInferredToReportedShift'],0],[['Active rivalry','rivalryInferredToReportedShift'],0,0,0],[['Active replay','replayInferredToReportedShift'],0,0,0]]
# # alignmentInfo=[['keys'],['keys'],['keys'],['keys']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # #-------and duration dependent
# # plotTitle='Active_rivalry_vs_replay_instantaneous_and_prolonged_switches_aligned_w_key'
# # sessionKinds=['Active rivalry','Active replay','Active replay']
# # regressors=[['reportedSwitches0duration_PupilOK','reportedSwitchesNon0duration_PupilOK','trialStarts'],['reportedSwitches0duration_PupilOK','reportedSwitchesNon0duration_PupilOK','trialStarts'],['physicalSwitches0duration_PupilOK','physicalSwitchesNon0duration_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0,1],[0,1],[0,1]]
# # timeShifts=[[0,0,0],[0,0,0],[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0]]
# # alignmentInfo=[['keys','keys'],['keys','keys'],['keys','keys']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_replay_vs_passive_replay_instantaneous_and_prolonged_switches_aligned_w_key'
# # sessionKinds=['Active replay','Active replay','Passive replay']
# # regressors=[['physicalSwitches0duration_PupilOK','physicalSwitchesNon0duration_PupilOK','trialStarts'],['reportedSwitches0duration_PupilOK','reportedSwitchesNon0duration_PupilOK','trialStarts'],['physicalSwitches0duration_PupilOK','physicalSwitchesNon0duration_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0,1],[0,1],[0,1]]
# # timeShifts=[[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0],[0,0,0],[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0]]
# # alignmentInfo=[['key','key'],['key','key'],['key','key']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_rivalry_and_replay_3_reported_switch_events_aligned_w_key'
# # sessionKinds=['Active rivalry','Active replay']
# # regressors=[['reportedSwitches0duration_PupilOK','reportedSwitchStarts_PupilOK','reportedSwitchEnds_PupilOK','trialStarts'],['reportedSwitches0duration_PupilOK','reportedSwitchStarts_PupilOK','reportedSwitchEnds_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0,1,2],[0,1,2]]
# # timeShifts=[[0,0,0,0],[0,0,0,0]]
# # alignmentInfo=[['key','key','key'],['key','key','key']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # #----and comparing effect of how you align it
# # plotTitle='Active_rivalry_reported_vs_inferred_aligned_either_way'
# # sessionKinds=['Active rivalry','Active rivalry','Active rivalry','Active rivalry']
# # regressors=[['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0],[0]]
# # timeShifts=[[['Active rivalry','rivalryReportedToInferredShift'],0],[0,0],[0,0],[['Active rivalry','rivalryInferredToReportedShift'],0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# # alignmentInfo=[['OKN'],['OKN'],['keys'],['keys']]
# # calculate_var_explained=False		#1: the one that is to be explained; 2: the one that does the explaining
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
# #
# # plotTitle='Active_replay_reported_vs_inferred_vs_physical_aligned_either_way'
# # sessionKinds=['Active replay','Active replay','Active replay','Active replay','Active replay','Active replay']
# # regressors=[['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['physicalSwitches_PupilOK','trialStarts'],['reportedSwitches_PupilOK','trialStarts'],['inferredSwitches_PupilOK','trialStarts'],['physicalSwitches_PupilOK','trialStarts']]
# # plottedRegressorIndices=[[0],[0],[0],[0],[0],[0]]
# # timeShifts=[[['Active replay','replayReportedToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0],[0,0],[['Active replay','replayInferredToReportedShift'],0],[['Active replay','replayPhysicalToReportedShift'],0]]
# # alignmentInfo=[['OKN'],['OKN'],['OKN'],['keys'],['keys'],['keys']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
#
# #-----------
# #define all the GLMS
#
# #first aligned to the inferred switch moment
# plotTitle='Active_rivalry_reported_vs_inferred_aligned_w_OKN'
# sessionKinds=['Active rivalry','Active rivalry']
# regressors=[['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts']]
# plottedRegressorIndices=[[0],[0]]
# timeShifts=[[['Active rivalry','rivalryReportedToInferredShift'],0],[0,0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# alignmentInfo=[['OKN'],['OKN']]	#information of regressors that trail what will be plotted, do not have to be filled out
# calculate_var_explained=[[1,0],[2,0]]		#1: the one that is to be explained; 2: the one that does the explaining
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_replay_reported_vs_inferred_vs_physical_aligned_w_OKN'
# sessionKinds=['Active replay','Active replay','Active replay']
# regressors=[['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts'],['physicalSwitches','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0]]
# timeShifts=[[['Active replay','replayReportedToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0]]
# alignmentInfo=[['OKN'],['OKN'],['OKN']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_vs_passive_rivalry_inferred_aligned_w_OKN'
# sessionKinds=['Active rivalry','Passive rivalry']
# regressors=[['inferredSwitches','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0],[0,1,2]]
# timeShifts=[[0,0],[0,0,0,0]]
# alignmentInfo=[['OKN','n/a'],['OKN','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_vs_passive_replay_physical_vs_inferred_aligned_w_OKN'
# sessionKinds=['Active replay','Active replay','Passive replay','Passive replay']
# regressors=[['physicalSwitches','trialStarts'],['inferredSwitches','trialStarts'],['physicalSwitches','probeReports','unreportedProbes','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0,1,2],[0]]
# timeShifts=[[['Active replay','replayPhysicalToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0,0,0],[0,0,0,0]]
# alignmentInfo=[['OKN'],['OKN','n/a'],['OKN','n/a','n/a'],['OKN','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_and_passive_rivalry_vs_replay_inferred_aligned_w_OKN'
# sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
# regressors=[['inferredSwitches','trialStarts'],['inferredSwitches','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0],[0]]
# timeShifts=[[0,0],[0,0],[0,0,0,0],[0,0,0,0]]
# alignmentInfo=[['OKN'],['OKN'],['OKN'],['OKN']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# #----and key press aligned
# plotTitle='Active_rivalry_reported_vs_inferred_aligned_w_key'
# sessionKinds=['Active rivalry','Active rivalry']
# regressors=[['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts']]
# plottedRegressorIndices=[[0],[0]]
# timeShifts=[[0,0],[['Active rivalry','rivalryInferredToReportedShift'],0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# alignmentInfo=[['keys'],['keys']]
# calculate_var_explained=[[1,0],[2,0]]		#1: the one that is to be explained; 2: the one that does the explaining
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_vs_passive_rivalry_inferred_aligned_w_key'
# sessionKinds=['Active rivalry','Passive rivalry']
# regressors=[['inferredSwitches','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0],[0,1,2]]
# timeShifts=[[['Active rivalry','rivalryInferredToReportedShift'],0],[['Active rivalry','rivalryInferredToReportedShift'],0,0,0]]
# alignmentInfo=[['keys'],['keys','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_and_passive_rivalry_vs_replay_inferred_aligned_w_key'
# sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
# regressors=[['inferredSwitches','trialStarts'],['inferredSwitches','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts'],['inferredSwitches','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0],[0]]
# timeShifts=[[['Active rivalry','rivalryInferredToReportedShift'],0],[['Active replay','replayInferredToReportedShift'],0],[['Active rivalry','rivalryInferredToReportedShift'],0,0,0],[['Active replay','replayInferredToReportedShift'],0,0,0]]
# alignmentInfo=[['keys'],['keys'],['keys'],['keys']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# #-------and duration dependent
# plotTitle='Active_rivalry_vs_replay_instantaneous_and_prolonged_switches_aligned_w_key'
# sessionKinds=['Active rivalry','Active replay','Active replay']
# regressors=[['reportedSwitches0duration','reportedSwitchesNon0duration','trialStarts'],['reportedSwitches0duration','reportedSwitchesNon0duration','trialStarts'],['physicalSwitches0duration','physicalSwitchesNon0duration','trialStarts']]
# plottedRegressorIndices=[[0,1],[0,1],[0,1]]
# timeShifts=[[0,0,0],[0,0,0],[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0]]
# alignmentInfo=[['keys','keys'],['keys','keys'],['keys','keys']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_replay_vs_passive_replay_instantaneous_and_prolonged_switches_aligned_w_key'
# sessionKinds=['Active replay','Active replay','Passive replay']
# regressors=[['physicalSwitches0duration','physicalSwitchesNon0duration','trialStarts'],['reportedSwitches0duration','reportedSwitchesNon0duration','trialStarts'],['physicalSwitches0duration','physicalSwitchesNon0duration','probeReports','unreportedProbes','trialStarts']]
# plottedRegressorIndices=[[0,1],[0,1],[0,1]]
# timeShifts=[[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0],[0,0,0],[['Active replay','replayPhysicalToReportedShift'],['Active replay','replayPhysicalToReportedShift'],0,0,0]]
# alignmentInfo=[['key','key'],['key','key'],['key','key']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_rivalry_and_replay_3_reported_switch_events_aligned_w_key'
# sessionKinds=['Active rivalry','Active replay']
# regressors=[['reportedSwitches0duration','reportedSwitchStarts','reportedSwitchEnds','trialStarts'],['reportedSwitches0duration','reportedSwitchStarts','reportedSwitchEnds','trialStarts']]
# plottedRegressorIndices=[[0,1,2],[0,1,2]]
# timeShifts=[[0,0,0,0],[0,0,0,0]]
# alignmentInfo=[['key','key','key'],['key','key','key']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# #----and comparing effect of how you align it
# plotTitle='Active_rivalry_reported_vs_inferred_aligned_either_way'
# sessionKinds=['Active rivalry','Active rivalry','Active rivalry','Active rivalry']
# regressors=[['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts'],['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0],[0]]
# timeShifts=[[['Active rivalry','rivalryReportedToInferredShift'],0],[0,0],[0,0],[['Active rivalry','rivalryInferredToReportedShift'],0]]			#each pair, if not a zero, is the condition name and the name of the time shift variable within that condition
# alignmentInfo=[['OKN'],['OKN'],['keys'],['keys']]
# calculate_var_explained=False		#1: the one that is to be explained; 2: the one that does the explaining
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# plotTitle='Active_replay_reported_vs_inferred_vs_physical_aligned_either_way'
# sessionKinds=['Active replay','Active replay','Active replay','Active replay','Active replay','Active replay']
# regressors=[['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts'],['physicalSwitches','trialStarts'],['reportedSwitches','trialStarts'],['inferredSwitches','trialStarts'],['physicalSwitches','trialStarts']]
# plottedRegressorIndices=[[0],[0],[0],[0],[0],[0]]
# timeShifts=[[['Active replay','replayReportedToInferredShift'],0],[0,0],[['Active replay','replayPhysicalToInferredShift'],0],[0,0],[['Active replay','replayInferredToReportedShift'],0],[['Active replay','replayPhysicalToReportedShift'],0]]
# alignmentInfo=[['OKN'],['OKN'],['OKN'],['keys'],['keys'],['keys']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# #-----------------
# #and then run the GLMS
#
# plotColors=['r','g','b','c', 'm', 'y', 'k']
# time0Index=int(abs(decoInterval[0])*newSampleRate)
# baselineIntervalIndices=[int((baselineStartEndTime-decoInterval[0])*newSampleRate) for baselineStartEndTime in baselineInterval]
#
# varExplainedTimeLimits=decoInterval
#
# for onePlotDict in plotDictList:
#
# 	individualsPerSessionKind=[[oneInfoDict['obs'] for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==thisSessionKind] for thisSessionKind in onePlotDict['sessionKinds']]
# 	individualsIncluded=list(reduce(set.intersection, map(set, individualsPerSessionKind)))
#
# 	allDecoResponses=[]
# 	allStErrs=[]
# 	allPerObsData=[]
#
# 	for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
#
# 		allDecoResponsesOneGLM=[]
#
# 		for oneObserver in individualsIncluded:
#
# 			thisInfoDict=[oneInfoDict for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==oneSessionKind and oneInfoDict['obs']==oneObserver][0]
# 			paddedPupilSamplesCleaned=thisInfoDict['paddedPupilSamplesCleaned']
# 			theseEventNames=onePlotDict['regressors'][glmIndex]
# 			theseEvents=[thisInfoDict[oneEventKind] for oneEventKind in theseEventNames]
#
# 			theseShifts=[0 if element==0 else [oneInfoDict for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==element[0] and oneInfoDict['obs']==oneObserver][0][element[1]] for element in onePlotDict['timeShifts'][glmIndex]]
# 			theseEvents=theseEvents+numpy.array(theseShifts)		#align the timing of reported and physical switches with that of the inferred ones
#
# 			eventTypesIndicesIncluded=[]
# 			for oneEventTypeIndex in range(len(theseEventNames)):
# 				if len(theseEvents[oneEventTypeIndex])>minNumEvs:
# 					eventTypesIndicesIncluded=eventTypesIndicesIncluded+[oneEventTypeIndex]
# 				else:
# 					print 'Event named '+theseEventNames[oneEventTypeIndex]+' excluded from '+onePlotDict['plotTitle']+' for observer '+oneObserver+' because insufficient events'
#
# 			theseEventsActuallyUsed=[theseEvents[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]
# 			theseEventNamesActuallyUsed=[theseEventNames[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]
#
# 			b = FIRDeconvolution.FIRDeconvolution(signal=sp.signal.decimate(paddedPupilSamplesCleaned, downSampleRate, 1),
# 			                         events=theseEventsActuallyUsed, event_names=theseEventNamesActuallyUsed, sample_frequency=newSampleRate,
# 			                         deconvolution_frequency=newSampleRate, deconvolution_interval=decoInterval,)
# 			b.create_design_matrix()
#
# 			if ridgeForrester:
# 				b.ridge_regress()
# 			else:
# 				b.regress()
#
# 			b.betas_for_events()
#
# 			decoResponsesOneGLMAndObsInCorrectOrder=[]
# 			for myExternalKey in theseEventNames:
# 				if not myExternalKey in theseEventNamesActuallyUsed:
# 					decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]]
# 				else:
# 					for thisIndex,thisName in enumerate(b.covariates.keys()):	#here use the internal .covariates.keys() because there is some sort of shuffling going on internally that determines the order of the regressors
# 						if thisName==myExternalKey:
# 							thisResponse=numpy.array(b.betas_per_event_type[thisIndex]).ravel()
# 							if subtractBaselineForPlots:
# 								thisResponse = thisResponse - numpy.average([thisResponse[baselineIndex] for baselineIndex in range(baselineIntervalIndices[0],baselineIntervalIndices[1])])	#baselined
# 							decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[thisResponse]
# 							break
#
# 			allDecoResponsesOneGLM=allDecoResponsesOneGLM+[decoResponsesOneGLMAndObsInCorrectOrder]
#
# 		theAverage=[]
# 		theStErr=[]
# 		thePerObsData=[]
# 		for regressorIndex in range(len(allDecoResponsesOneGLM[0])):
# 			onlyIncludedObservers=[allDecoResponsesOneGLM[obsIndex][regressorIndex] for obsIndex in range(len(allDecoResponsesOneGLM)) if not (min(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1 and max(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1)]
# 			if onlyIncludedObservers==[]:
# 				thisAverage=[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]
# 			else:
# 				thisAverage=numpy.average(onlyIncludedObservers,0)
# 				thisStErr=numpy.std(onlyIncludedObservers,0)/float(numpy.sqrt(len(onlyIncludedObservers)))
# 			theAverage=theAverage+[thisAverage]
# 			theStErr=theStErr+[thisStErr]		#theStErr is all the across-obs st Errs (one per regressor) within this GLM
# 			thePerObsData=thePerObsData+[onlyIncludedObservers]	#thePerObsData is, for this GLM, all the individual-observer data for included observers only. Nesting is: all regressors, all included observers, all timepoints
#
# 		#allDecoResponsesOneGLM=allDecoResponsesOneGLM+[numpy.average(allDecoResponsesOneGLM,0)]
# 		allDecoResponsesOneGLM=allDecoResponsesOneGLM+[theAverage]	#add theAverage as if it's just another participant
# 		allDecoResponses=allDecoResponses+[allDecoResponsesOneGLM]
# 		allStErrs=allStErrs+[theStErr]		#allStErrs is bunch of theStErr's, one for each GLM
# 		allPerObsData=allPerObsData+[thePerObsData]		#allPerObsData is bunch of thePerObsData's, one for each GLM. So nesting is: GLM, regressor, observer (only included ones), timepoint. It's very similar to allDecoResponsesOneGLM but nested in a different order and with individual observers removed if their data didn't include a particular regressor.
#
# 	x = numpy.linspace(decoInterval[0],decoInterval[1], len(allDecoResponses[0][0][0]))
#
# 	rVals=[]
# 	if onePlotDict['calculateVarExplained']:
#
# 		toBeExplainedIndex=numpy.where(numpy.array(onePlotDict['calculateVarExplained'])==1)
# 		explainerIndex=numpy.where(numpy.array(onePlotDict['calculateVarExplained'])==2)
#
# 		toBeExplainedName=onePlotDict['sessionKinds'][int(toBeExplainedIndex[0])]+'; '+onePlotDict['regressors'][int(toBeExplainedIndex[0])][int(toBeExplainedIndex[1])]
# 		explainerName=onePlotDict['sessionKinds'][int(explainerIndex[0])]+'; '+onePlotDict['regressors'][int(explainerIndex[0])][int(explainerIndex[1])]
#
# 		for obsIndex in range(len(allDecoResponses[0])):
#
# 			toBeExplained=allDecoResponses[int(toBeExplainedIndex[0])][obsIndex][int(toBeExplainedIndex[1])]
# 			toBeExplained=[thisElement for thisIndex,thisElement in enumerate(toBeExplained) if (x[thisIndex]>varExplainedTimeLimits[0]) and (x[thisIndex]<varExplainedTimeLimits[1])]
#
# 			explainer=allDecoResponses[int(explainerIndex[0])][obsIndex][int(explainerIndex[1])]
# 			explainer=[thisElement for thisIndex,thisElement in enumerate(explainer) if (x[thisIndex]>varExplainedTimeLimits[0]) and (x[thisIndex]<varExplainedTimeLimits[1])]
#
# 			rVals=rVals+[sp.stats.pearsonr(toBeExplained, explainer)[0]]
#
# 			# #meanToBeExplained=numpy.average(toBeExplained)
# 			# #meanExplainer=numpy.average(explainer)
# 			#
# 			# #sseReAverage=sum([pow(element-meanToBeExplained,2) for element in toBeExplained])
# 			# #sseReExplainer=sum([pow((toBeExplained[index]-meanToBeExplained)-(explainer[index]-meanExplainer),2) for index in range(len(toBeExplained))])
# 			# sseReAverage=sum([pow(element,2) for element in toBeExplained])
# 			# sseReExplainer=sum([pow(toBeExplained[index]-explainer[index],2) for index in range(len(toBeExplained))])
# 			# ssePropExplained=(sseReAverage-sseReExplainer)/sseReAverage
# 			#
# 			# ssePropsExplained=ssePropsExplained+[ssePropExplained]
#
# 	individualsIncludedPlusAverage=individualsIncluded+['Average']
# 	f = pl.figure(figsize = (35,35))
# 	for observerIndex in range(len(individualsIncluded)+1):
#
# 		s=f.add_subplot(5,6,observerIndex+1)
# 		colorCounter=0
# 		for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 			theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
# 			for regressorIndex in theseRegressorIndices:
# 				y=allDecoResponses[glmIndex][observerIndex][regressorIndex]
# 				pl.plot(x, y, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
# 				colorCounter=colorCounter+1
#
# 		pl.xlabel('Time from event (s)')
# 		pl.ylabel('Pupil size')
# 		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 		#sn.despine(offset=10)
# 		if rVals:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex]+'. r \''+toBeExplainedName+'\'\nvs  \''+explainerName+'\': '+str(round(100*rVals[observerIndex])/100))
# 		else:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex])
#
# 	pl.legend(loc=2)
#
# 	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
# 	colorCounter=0
# 	for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
# 		for regressorIndex in theseRegressorIndices:
# 			y=allDecoResponses[glmIndex][-1][regressorIndex]		#-1 will be the across-obs average
# 			stErrs=allStErrs[glmIndex][regressorIndex]
#
# 			tTestpValsVs0=[ttest_1samp([allPerObsData[glmIndex][regressorIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(len(x))]
#
# 			pl.errorbar(x, y, yerr=stErrs, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
#
# 			xForSignificantOnes=[]
# 			yForSignificantOnes=[]
# 			for candidateIndex in range(len(x)):
# 				if tTestpValsVs0[candidateIndex]<.01:
# 					xForSignificantOnes=xForSignificantOnes+[x[candidateIndex]]
# 					yForSignificantOnes=yForSignificantOnes+[y[candidateIndex]]
#
# 			pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)
#
# 			colorCounter=colorCounter+1
#
# 	pl.xlabel('Time from event (s)')
# 	pl.ylabel('Pupil size')
# 	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 	#sn.despine(offset=10)
# 	s.set_title('Average plus error bars')
#
# 	if ridgeForrester:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_ridge.pdf')
# 	else:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'.pdf')
#
# #--------------------------
# #And now define more GLMS, now while including saccades and blinks in the GLM rather than regressing them out beforehand
# plotDictList=[]
#
# plotTitle='Active_vs_passive_rivalry_inferred_aligned_w_OKN'
# sessionKinds=['Active rivalry','Passive rivalry']
# regressors=[['inferredSwitches','trialStarts','blinks','saccades'],['inferredSwitches','probeReports','unreportedProbes','trialStarts','blinks','saccades']]
# plottedRegressorIndicesA=[[0],[0,1,2]]
# timeShifts=[[0,0,0,0],[0,0,0,0,0,0]]
# alignmentInfo=[['OKN','n/a','n/a','n/a'],['OKN','n/a','n/a','n/a','n/a','n/a']]
# plottedRegressorIndicesB=[[2,3],[4,5]]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'sessionKinds':sessionKinds,'regressors':regressors,'plottedRegressorIndicesA':plottedRegressorIndicesA,'plottedRegressorIndicesB':plottedRegressorIndicesB,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo}]
#
# #-----------------
# #and then run the GLMS
#
# plotColors=['r','g','b','c', 'm', 'y', 'k',(0.3, 0.3, 0.3)]
# time0Index=int(abs(decoInterval[0])*newSampleRate)
# baselineIntervalIndices=[int((baselineStartEndTime-decoInterval[0])*newSampleRate) for baselineStartEndTime in baselineInterval]
#
# varExplainedTimeLimits=decoInterval
#
# for onePlotDict in plotDictList:
#
# 	individualsPerSessionKind=[[oneInfoDict['obs'] for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==thisSessionKind] for thisSessionKind in onePlotDict['sessionKinds']]
# 	individualsIncluded=list(reduce(set.intersection, map(set, individualsPerSessionKind)))
#
# 	allDecoResponses=[]
# 	allStErrs=[]
# 	allPerObsData=[]
#
# 	for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
#
# 		allDecoResponsesOneGLM=[]
#
# 		for oneObserver in individualsIncluded:
#
# 			thisInfoDict=[oneInfoDict for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==oneSessionKind and oneInfoDict['obs']==oneObserver][0]
# 			paddedPupilSamplesCleaned=thisInfoDict['paddedPupilSamplesSaccBlinksNotRemoved']
# 			theseEventNames=onePlotDict['regressors'][glmIndex]
# 			theseEvents=[thisInfoDict[oneEventKind] for oneEventKind in theseEventNames]
#
# 			theseShifts=[0 if element==0 else [oneInfoDict for oneInfoDict in allInfoDictList if oneInfoDict['sessionKind']==element[0] and oneInfoDict['obs']==oneObserver][0][element[1]] for element in onePlotDict['timeShifts'][glmIndex]]
# 			theseEvents=theseEvents+numpy.array(theseShifts)		#align the timing of reported and physical switches with that of the inferred ones
#
# 			eventTypesIndicesIncluded=[]
# 			for oneEventTypeIndex in range(len(theseEventNames)):
# 				if len(theseEvents[oneEventTypeIndex])>minNumEvs:
# 					eventTypesIndicesIncluded=eventTypesIndicesIncluded+[oneEventTypeIndex]
# 				else:
# 					print 'Event named '+theseEventNames[oneEventTypeIndex]+' excluded from '+plotTitle+' for observer '+oneObserver+' because insufficient events'
#
# 			theseEventsActuallyUsed=[theseEvents[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]
# 			theseEventNamesActuallyUsed=[theseEventNames[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]
#
# 			#----make a few plots to see what you're putting into the regressor----
#
# 			if observerIndex==0:
#
# 				colors=['b','g','r','c','m','y','k',(.5,.5,.5)]
# 				symbols=['o','x','+']
# 				colorSymbolCombis=[]
# 				for oneColor in colors:
# 					colorSymbolCombis=colorSymbolCombis+[[oneColor,oneSymbol] for oneSymbol in symbols]
#
# 				print("THE THIRD ARGUMENT OF THIS DECIMATE IS STILL NONSENSE, BUT I'M NOT USING THIS CODE FOR ANYTHING RIGHT NOW: USING NIDECONV ON CONCATENATED DATA")
# 				downSampledSignal=sp.signal.decimate(paddedPupilSamplesCleaned, downSampleRate, 1)
#
# 				startTime_s=0
# 				endTime_s=timePerPlot_s
#
# 				while endTime_s<min([maxTimepointForReviewPlots_s,numpy.floor(float(len(downSampledSignal))/float(newSampleRate))]):
#
# 					fig = pl.figure(figsize = (25,15))
# 					s = fig.add_subplot(1,1,1)
# 					s.set_title(onePlotDict['plotTitle']+', '+oneSessionKind+', '+oneObserver)
#
# 					y_DataForPlot=downSampledSignal[int(startTime_s*newSampleRate):int(endTime_s*newSampleRate)]
# 					x_DataForPlot=[startTime_s+float(xAxisIndex)/newSampleRate for xAxisIndex in range(len(y_DataForPlot))]
#
# 					if len(y_DataForPlot>0):
#
# 						minY=min(y_DataForPlot)
# 						maxY=max(y_DataForPlot)
#
# 						pl.plot(x_DataForPlot,y_DataForPlot, color = 'k', linewidth=1.2, label='pupil')
#
# 						for eventIndexForPlot, oneEventDataForPlot in enumerate(theseEventsActuallyUsed):
#
# 							theseEventDataForPlot=[element for element in oneEventDataForPlot if element>=startTime_s and element<=endTime_s]
#
# 							if len(theseEventDataForPlot)>0:
#
# 								[pl.plot([element,element],[minY,maxY], color = colorSymbolCombis[numpy.mod(eventIndexForPlot,len(plotColors))][0], linewidth=.5) for element in theseEventDataForPlot]
# 								pl.scatter(theseEventDataForPlot, [maxY for wimpie in theseEventDataForPlot], color = colorSymbolCombis[numpy.mod(eventIndexForPlot,len(plotColors))][0], marker = colorSymbolCombis[numpy.mod(eventIndexForPlot,len(plotColors))][1], label=theseEventNamesActuallyUsed[eventIndexForPlot])
#
# 					pl.legend()
# 					fileNameWithPotentiallySpaces=myPath+figuresSubFolder+'/'+'regressed_data_example_'+onePlotDict['plotTitle']+'_'+oneSessionKind+'_'+oneObserver+'_'+str(startTime_s)+'.pdf'
# 					pl.savefig('_'.join(fileNameWithPotentiallySpaces.split()))
# 					pl.close()
#
# 					startTime_s=endTime_s
# 					endTime_s=startTime_s+timePerPlot_s
#
# 			#-----------------------------------------------------------------------
#
# 			print("THE THIRD ARGUMENT OF THIS DECIMATE IS STILL NONSENSE, BUT I'M NOT USING THIS CODE FOR ANYTHING RIGHT NOW: USING NIDECONV ON CONCATENATED DATA")
# 			b = FIRDeconvolution.FIRDeconvolution(signal=sp.signal.decimate(paddedPupilSamplesCleaned, downSampleRate, 1),
# 			                         events=theseEventsActuallyUsed, event_names=theseEventNamesActuallyUsed, sample_frequency=newSampleRate,
# 			                         deconvolution_frequency=newSampleRate, deconvolution_interval=decoInterval,)
# 			b.create_design_matrix()
#
# 			if ridgeForrester:
# 				b.ridge_regress()
# 			else:
# 				b.regress()
#
# 			b.betas_for_events()
#
# 			decoResponsesOneGLMAndObsInCorrectOrder=[]
# 			for myExternalKey in theseEventNames:
# 				if not myExternalKey in theseEventNamesActuallyUsed:
# 					decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]]
# 				else:
# 					for thisIndex,thisName in enumerate(b.covariates.keys()):	#here use the internal .covariates.keys() because there is some sort of shuffling going on internally that determines the order of the regressors
# 						if thisName==myExternalKey:
# 							thisResponse=numpy.array(b.betas_per_event_type[thisIndex]).ravel()
# 							if subtractBaselineForPlots:
# 								thisResponse = thisResponse - numpy.average([thisResponse[baselineIndex] for baselineIndex in range(baselineIntervalIndices[0],baselineIntervalIndices[1])])	#baselined
# 							decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[thisResponse]
# 							break
#
# 			allDecoResponsesOneGLM=allDecoResponsesOneGLM+[decoResponsesOneGLMAndObsInCorrectOrder]
#
# 		theAverage=[]
# 		theStErr=[]
# 		thePerObsData=[]
# 		for regressorIndex in range(len(allDecoResponsesOneGLM[0])):
# 			onlyIncludedObservers=[allDecoResponsesOneGLM[obsIndex][regressorIndex] for obsIndex in range(len(allDecoResponsesOneGLM)) if not (min(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1 and max(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1)]
# 			if onlyIncludedObservers==[]:
# 				thisAverage=[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]
# 			else:
# 				thisAverage=numpy.average(onlyIncludedObservers,0)
# 				thisStErr=numpy.std(onlyIncludedObservers,0)/float(numpy.sqrt(len(onlyIncludedObservers)))
# 			theAverage=theAverage+[thisAverage]
# 			theStErr=theStErr+[thisStErr]		#theStErr is all the across-obs st Errs (one per regressor) within this GLM
# 			thePerObsData=thePerObsData+[onlyIncludedObservers]	#thePerObsData is, for this GLM, all the individual-observer data for included observers only. Nesting is: all regressors, all included observers, all timepoints
#
# 		#allDecoResponsesOneGLM=allDecoResponsesOneGLM+[numpy.average(allDecoResponsesOneGLM,0)]
# 		allDecoResponsesOneGLM=allDecoResponsesOneGLM+[theAverage]	#add theAverage as if it's just another participant
# 		allDecoResponses=allDecoResponses+[allDecoResponsesOneGLM]
# 		allStErrs=allStErrs+[theStErr]		#allStErrs is bunch of theStErr's, one for each GLM
# 		allPerObsData=allPerObsData+[thePerObsData]		#allPerObsData is bunch of thePerObsData's, one for each GLM. So nesting is: GLM, regressor, observer (only included ones), timepoint. It's very similar to allDecoResponsesOneGLM but nested in a different order and with individual observers removed if their data didn't include a particular regressor.
#
# 	x = numpy.linspace(decoInterval[0],decoInterval[1], len(allDecoResponses[0][0][0]))
#
# 	rVals=[]
# 	if onePlotDict['calculateVarExplained']:
#
# 		toBeExplainedIndex=numpy.where(numpy.array(onePlotDict['calculateVarExplained'])==1)
# 		explainerIndex=numpy.where(numpy.array(onePlotDict['calculateVarExplained'])==2)
#
# 		toBeExplainedName=onePlotDict['sessionKinds'][int(toBeExplainedIndex[0])]+'; '+onePlotDict['regressors'][int(toBeExplainedIndex[0])][int(toBeExplainedIndex[1])]
# 		explainerName=onePlotDict['sessionKinds'][int(explainerIndex[0])]+'; '+onePlotDict['regressors'][int(explainerIndex[0])][int(explainerIndex[1])]
#
# 		for obsIndex in range(len(allDecoResponses[0])):
#
# 			toBeExplained=allDecoResponses[int(toBeExplainedIndex[0])][obsIndex][int(toBeExplainedIndex[1])]
# 			toBeExplained=[thisElement for thisIndex,thisElement in enumerate(toBeExplained) if (x[thisIndex]>varExplainedTimeLimits[0]) and (x[thisIndex]<varExplainedTimeLimits[1])]
#
# 			explainer=allDecoResponses[int(explainerIndex[0])][obsIndex][int(explainerIndex[1])]
# 			explainer=[thisElement for thisIndex,thisElement in enumerate(explainer) if (x[thisIndex]>varExplainedTimeLimits[0]) and (x[thisIndex]<varExplainedTimeLimits[1])]
#
# 			rVals=rVals+[sp.stats.pearsonr(toBeExplained, explainer)[0]]
#
# 			# #meanToBeExplained=numpy.average(toBeExplained)
# 			# #meanExplainer=numpy.average(explainer)
# 			#
# 			# #sseReAverage=sum([pow(element-meanToBeExplained,2) for element in toBeExplained])
# 			# #sseReExplainer=sum([pow((toBeExplained[index]-meanToBeExplained)-(explainer[index]-meanExplainer),2) for index in range(len(toBeExplained))])
# 			# sseReAverage=sum([pow(element,2) for element in toBeExplained])
# 			# sseReExplainer=sum([pow(toBeExplained[index]-explainer[index],2) for index in range(len(toBeExplained))])
# 			# ssePropExplained=(sseReAverage-sseReExplainer)/sseReAverage
# 			#
# 			# ssePropsExplained=ssePropsExplained+[ssePropExplained]
#
# 	individualsIncludedPlusAverage=individualsIncluded+['Average']
# 	f = pl.figure(figsize = (35,35))
# 	for observerIndex in range(len(individualsIncluded)+1):
#
# 		s=f.add_subplot(5,6,observerIndex+1)
# 		colorCounter=0
# 		for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 			theseRegressorIndices=onePlotDict['plottedRegressorIndicesA'][glmIndex]
# 			for regressorIndex in theseRegressorIndices:
# 				y=allDecoResponses[glmIndex][observerIndex][regressorIndex]
# 				pl.plot(x, y, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
#
# 				colorCounter=colorCounter+1
#
# 		pl.xlabel('Time from event (s)')
# 		pl.ylabel('Pupil size')
# 		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 		#sn.despine(offset=10)
# 		if rVals:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex]+'. r \''+toBeExplainedName+'\'\nvs  \''+explainerName+'\': '+str(round(100*rVals[observerIndex])/100))
# 		else:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex])
#
# 	pl.legend(loc=2)
#
# 	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
# 	colorCounter=0
# 	for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 		theseRegressorIndices=onePlotDict['plottedRegressorIndicesA'][glmIndex]
# 		for regressorIndex in theseRegressorIndices:
# 			y=allDecoResponses[glmIndex][-1][regressorIndex]		#-1 will be the across-obs average
# 			stErrs=allStErrs[glmIndex][regressorIndex]
#
# 			tTestpValsVs0=[ttest_1samp([allPerObsData[glmIndex][regressorIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(len(x))]
#
# 			pl.errorbar(x, y, yerr=stErrs, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
#
# 			xForSignificantOnes=[]
# 			yForSignificantOnes=[]
# 			for candidateIndex in range(len(x)):
# 				if tTestpValsVs0[candidateIndex]<.01:
# 					xForSignificantOnes=xForSignificantOnes+[x[candidateIndex]]
# 					yForSignificantOnes=yForSignificantOnes+[y[candidateIndex]]
#
# 			pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)
#
# 			colorCounter=colorCounter+1
#
# 	pl.xlabel('Time from event (s)')
# 	pl.ylabel('Pupil size')
# 	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 	#sn.despine(offset=10)
# 	s.set_title('Average plus error bars')
#
# 	if ridgeForrester:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_includedSaccsAndBlinks_ridge_A.pdf')
# 	else:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_includedSaccsAndBlinks_A.pdf')
#
# 	f = pl.figure(figsize = (35,35))
# 	for observerIndex in range(len(individualsIncluded)+1):
#
# 		s=f.add_subplot(5,6,observerIndex+1)
# 		colorCounter=0
# 		for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 			theseRegressorIndices=onePlotDict['plottedRegressorIndicesB'][glmIndex]
# 			for regressorIndex in theseRegressorIndices:
# 				y=allDecoResponses[glmIndex][observerIndex][regressorIndex]
# 				pl.plot(x, y, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
#
# 				colorCounter=colorCounter+1
#
# 		pl.xlabel('Time from event (s)')
# 		pl.ylabel('Pupil size')
# 		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 		#sn.despine(offset=10)
# 		if rVals:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex]+'. r \''+toBeExplainedName+'\'\nvs  \''+explainerName+'\': '+str(round(100*rVals[observerIndex])/100))
# 		else:
# 			s.set_title(individualsIncludedPlusAverage[observerIndex])
#
# 	pl.legend(loc=2)
#
# 	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
# 	colorCounter=0
# 	for glmIndex,oneSessionKind in enumerate(onePlotDict['sessionKinds']):
# 		theseRegressorIndices=onePlotDict['plottedRegressorIndicesB'][glmIndex]
# 		for regressorIndex in theseRegressorIndices:
# 			y=allDecoResponses[glmIndex][-1][regressorIndex]		#-1 will be the across-obs average
# 			stErrs=allStErrs[glmIndex][regressorIndex]
#
# 			tTestpValsVs0=[ttest_1samp([allPerObsData[glmIndex][regressorIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(len(x))]
#
# 			pl.errorbar(x, y, yerr=stErrs, color=plotColors[colorCounter], label=oneSessionKind+', '+onePlotDict['regressors'][glmIndex][regressorIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex])
#
# 			xForSignificantOnes=[]
# 			yForSignificantOnes=[]
# 			for candidateIndex in range(len(x)):
# 				if tTestpValsVs0[candidateIndex]<.01:
# 					xForSignificantOnes=xForSignificantOnes+[x[candidateIndex]]
# 					yForSignificantOnes=yForSignificantOnes+[y[candidateIndex]]
#
# 			pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)
#
# 			colorCounter=colorCounter+1
#
# 	pl.xlabel('Time from event (s)')
# 	pl.ylabel('Pupil size')
# 	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
# 	#sn.despine(offset=10)
# 	s.set_title('Average plus error bars')
#
# 	if ridgeForrester:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_includedSaccsAndBlinks_ridge_B.pdf')
# 	else:
# 		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+'_'+str(decoInterval[0])+'_'+str(decoInterval[1])+'_'+str(downSampleRate)+'_includedSaccsAndBlinks_B.pdf')
