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
import pickle
from scipy.stats import ttest_1samp
import copy
import math

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
import nideconv

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

downSampleFactors=[10,10]	#downsample in multiple steps if total downSampleFactor>13. Recommended for decimate function
downSampleFactor=downSampleFactors[0]
for additionalFactor in downSampleFactors[1:]:
	downSampleFactor=downSampleFactor*additionalFactor
	
newSampleRate=basicSampleRate/downSampleFactor

downsampledSamplesPerTrial=int(trialDurS*float(newSampleRate))

decoInterval=[-3.5,6.5]	#for pupil data; not behavioral data. was [-3.5,5.]
decoIntervalSacc=[-.5,4.5]	#was [-1.5,6.5]
decoIntervalBlink=[-.5,7.5]	#was [-1.5,6.5]

# baselineInterval=[-3.5,-3.]
# subtractBaselineForPlots=False
# ridgeForrester=False

numBasisFunctionsIncOffset=int((decoInterval[1]-decoInterval[0])*2.+1.)	#used to be 25. Now adaptively whatever is 1 Hz
numBasisFunctionsIncOffsetSacc=int((decoIntervalSacc[1]-decoIntervalSacc[0])*2.+1.)
numBasisFunctionsIncOffsetBlink=int((decoIntervalBlink[1]-decoIntervalBlink[0])*2.+1.)

minNumEvs=5 		#if a regressor has fewer than this number of events, then don't try.

print('DONT FORGET THAT CONCATENATEDDICTLIST IS READ DIRECTLY IF AVAILABLE, SO ANY MORE RECENT CHANGES TO DICTLIST WILL NOT BE REFLECTED IN THIS ANALYSIS. HAVE TO OVERWRITE CONCATENATEDDICTLIST FOR THAT')

sessionKinds=['Active rivalry','Active replay','Passive rivalry','Passive replay']
timeSeriesString='paddedPupilSamplesSaccBlinksNotRemoved'
regressorStrings=['trialStarts','trialEnds','saccades','blinks',
'reportedSwitches','reportedSwitches0duration','reportedSwitchesNon0duration',
'physicalSwitches','physicalSwitches0duration','physicalSwitchesNon0duration',
'inferredSwitches','inferredSwitches0duration','inferredSwitchesNon0duration','inferredSwitchesNoProbeInSubsequentPerc','inferredSwitchesProbeInSubsequentPerc',
'inferredSwitchesNoBlink','inferredSwitchesBlink',
'shownProbes','probeReports','unreportedProbes',
'reportedSwitchStarts','reportedSwitchEnds',
'physicalSwitchStarts','physicalSwitchEnds',
'physicalSwitchDurations','reportedSwitchDurations']

covariateListsPre=[['inferredSwitches',['inferredSwitches'],'covInfSwitchInfIntervalPre'],		#which regressor supplies the events; which regressors supply the preceding interval (the last moment out of all regressors is chosen); what will be the name under which this covariate is stored
['physicalSwitches',['physicalSwitches'],'covPhysSwitchPhysIntervalPre'],
['physicalSwitches0duration',['physicalSwitches'],'covPhysSwitch0durPhysIntervalPre'],
['physicalSwitches0duration',['physicalSwitchEnds','physicalSwitches0duration'],'covPhysSwitch0durPhysEndIntervalPre'],
['probeReports',['probeReports'],'covProbeReportReportIntervalPre'],
['reportedSwitches',['reportedSwitches'],'covRepSwitchRepIntervalPre'],
['reportedSwitches0duration',['reportedSwitches'],'covRepSwitch0durRepIntervalPre'],
['reportedSwitches0duration',['reportedSwitchEnds','reportedSwitches0duration'],'covRepSwitch0durRepEndIntervalPre'],
['inferredSwitches0duration',['inferredSwitches'],'covInfSwitch0durInfIntervalPre']	,
['inferredSwitchesNoProbeInSubsequentPerc',['inferredSwitches'],'covInfSwitchNoProbeSubsqInfIntervalPre'],						#for inferred switches the '0duration' variant is only available for passive replay, where it is based on duration of matching physical switch (in code titled '1a_...')
['inferredSwitchesNoBlink',['inferredSwitches'],'covInfSwitchNoBlinkInfIntervalPre'],
['probeReports',['probeReports'],'covProbeRepProbeRepIntervalPre']
]

covariateListsPost=[['inferredSwitches',['inferredSwitches'],'covInfSwitchInfIntervalPost'],		#which regressor supplies the events; which regressors supply the following interval (the first moment out of all regressors is chosen); what will be the name under which this covariate is stored
['physicalSwitches',['physicalSwitches'],'covPhysSwitchPhysIntervalPost'],
['physicalSwitches0duration',['physicalSwitches'],'covPhysSwitch0durPhysIntervalPost'],
['physicalSwitches0duration',['physicalSwitchStarts','physicalSwitches0duration'],'covPhysSwitch0durPhysStartIntervalPost'],
['probeReports',['probeReports'],'covProbeReportReportIntervalPost'],
['reportedSwitches',['reportedSwitches'],'covRepSwitchRepIntervalPost'],
['reportedSwitches0duration',['reportedSwitches'],'covRepSwitch0durRepIntervalPost'],
['reportedSwitches0duration',['reportedSwitchStarts','reportedSwitches0duration'],'covRepSwitch0durRepStartIntervalPost'],
['inferredSwitches0duration',['inferredSwitches'],'covInfSwitch0durInfIntervalPost'],
['inferredSwitchesNoProbeInSubsequentPerc',['inferredSwitches'],'covInfSwitchNoProbeSubsqInfIntervalPost'],
['inferredSwitchesNoBlink',['inferredSwitches'],'covInfSwitchNoBlinkInfIntervalPost'],
['probeReports',['probeReports'],'covProbeRepProbeRepIntervalPost']
]

covariateListsOther=[['reportedSwitches','reportedSwitchDurations','covRepSwitchRepSwitchdur'],					#which regressor supplies the events; which regressor supplies the covariant variable directly; what will be the name under which this covariate is stored
['physicalSwitches','physicalSwitchDurations','covPhysSwitchPhysSwitchdur'],
]

shiftStrings=['rivalryInferredToReportedShift',
'rivalryReportedToInferredShift',
'replayReportedToInferredShift',
'replayInferredToReportedShift',
'replayPhysicalToReportedShift',
'replayPhysicalToInferredShift']

numSections=3
splitLists=[['inferredSwitches','covInfSwitchInfIntervalPre',numSections],		#split the regressor in first member of a triplet according to the ascending order of the covariate in the second member, making a total number of sections determined by the third member
['physicalSwitches','covPhysSwitchPhysIntervalPre',numSections],
['reportedSwitches','covRepSwitchRepIntervalPre',numSections],
['inferredSwitches','covInfSwitchInfIntervalPost',numSections],
['physicalSwitches','covPhysSwitchPhysIntervalPost',numSections],
['reportedSwitches','covRepSwitchRepIntervalPost',numSections],
['reportedSwitches0duration','covRepSwitch0durRepIntervalPost',numSections],
['physicalSwitches0duration','covPhysSwitch0durPhysIntervalPre',numSections],
['inferredSwitches0duration','covInfSwitch0durInfIntervalPre',numSections],
['inferredSwitchesNoProbeInSubsequentPerc','covInfSwitchNoProbeSubsqInfIntervalPost',numSections]
]

#about the joining stuff: there is automatically joining of all like-named regressors/covariates across all conditions. But these designated lists are if you want to join unlike-named things, or if you want to join across only some conditions (e.g. rivalry only)
joinSets=[[['reportedSwitches0duration_Any','reportedSwitchStarts_Any','reportedSwitchEnds_Any','probeReports_Any'],'keyPress_Any'],
[['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry'],'inferredRivalrySwitches_Any'],
[['inferredSwitches_Active replay','inferredSwitches_Passive replay'],'inferredReplaySwitches_Any'],
]

joinSetsCovariates=[['covInfSwitchInfIntervalPre',['Active rivalry','Passive rivalry'],'Any rivalry'],	#This is for joining other than automatic joining of all like-named covariates across all conditions. Interpretation: covariate type to be joined, list of session type to include in the joined set, suffix that the result should get. z-scoring has been done within condition, if it is done at all.
['covInfSwitchInfIntervalPost',['Active rivalry','Passive rivalry'],'Any rivalry'],
['covInfSwitchInfIntervalPre',['Active replay','Passive replay'],'Any replay'],
['covInfSwitchInfIntervalPost',['Active replay','Passive replay'],'Any replay'],
['covPhysSwitchPhysIntervalPre',['Active replay','Passive replay'],'Any replay'],		#there is already an 'Any' of this, automatically created when all physicalSwitches regressors were joined
['covPhysSwitchPhysIntervalPost',['Active replay','Passive replay'],'Any replay'],
['covInfSwitchInfIntervalPre',['Active rivalry','Active replay'],'Any active'],
['covInfSwitchInfIntervalPost',['Active rivalry','Active replay'],'Any active'],
['covInfSwitchInfIntervalPre',['Passive rivalry','Passive replay'],'Any passive'],
['covInfSwitchInfIntervalPost',['Passive rivalry','Passive replay'],'Any passive'],
['covRepSwitchRepIntervalPre',['Active rivalry','Active replay'],'Any active'],
['covRepSwitchRepIntervalPost',['Active rivalry','Active replay'],'Any active'],
['covPhysSwitch0durPhysIntervalPre',['Active replay','Passive replay'],'Any replay'],
['covPhysSwitch0durPhysIntervalPost',['Active replay','Passive replay'],'Any replay'],
['covPhysSwitch0durPhysEndIntervalPre',['Active replay','Passive replay'],'Any replay'],
['covPhysSwitch0durPhysStartIntervalPost',['Active replay','Passive replay'],'Any replay'],
['covPhysSwitchPhysSwitchdur',['Active replay','Passive replay'],'Any replay'],
['covRepSwitchRepSwitchdur',['Active rivalry','Active replay'],'Any active'],
['covRepSwitch0durRepIntervalPre',['Active rivalry','Active replay'],'Any active'],
['covRepSwitch0durRepIntervalPost',['Active rivalry','Active replay'],'Any active']
]
#Beware: if you end up using joint regressors produced by joinSets along with joint covariates produced by joinSetsCovariates, you'll have to make sure that the order in argument 1 matches across both 'joinSets' variables.

joinSetsSplitLists=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre',['Active rivalry','Passive rivalry'],'Any rivalry'],	#interpretation as joinSetsCovariates. Splitting has been done within conditions; then joining.
['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost',['Active rivalry','Passive rivalry'],'Any rivalry'],
['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre',['Active replay','Passive replay'],'Any replay'],
['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost',['Active replay','Passive replay'],'Any replay'],
]

allInfoDictListFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_allInfoDictList' in element]
uniqueObservers=[fileName.split('_')[0] for fileName in allInfoDictListFilesPresent]

uniqueObservers=[thisObs for thisObs in uniqueObservers if not thisObs in excludedObs] 

allConcatenatedDictListFilesPresent=[element for element in os.listdir(myPath+miscStuffSubFolder+'/') if '_concatenatedDictList' in element]

concatenatedDictList=[]
for oneObs in uniqueObservers:
	
	if oneObs+'_concatenatedDictList' in allConcatenatedDictListFilesPresent:# and False:
		
		print 'Directly loading concatenated data for '+oneObs+' because already present.'
		with open(myPath+miscStuffSubFolder+'/'+oneObs+'_concatenatedDictList', "r") as f:
			concatenatedDict=pickle.load(f)
			
		allSPLITBYkeys=[element for element in concatenatedDict.keys() if 'SPLITBY' in element and str(numSections)+'_' in element]
		
		if len(allSPLITBYkeys)>0:
			print('Warning! Currently the code says to split into '+str(numSections)+' sections. But the following dictionary keys suggest the loaded data contain more:')
			for element in allSPLITBYkeys:
				print element
			print('Is that ok?')
			print('Please terminate and re-run after removing concatenatedDictList items if not!!')
		
	else:	
		
		print 'Computing concatenated data for '+oneObs+' because concatenated file not yet present.'
		with open(myPath+miscStuffSubFolder+'/'+oneObs+'_allInfoDictList', "r") as f:
			oneDictList=pickle.load(f)
		
		totalOffset=0.	#how much do the regressors have to be shifted by (in seconds) to align with the concatenated data
	
		concatenatedDict={'obs':oneDictList[0]['obs']}
		concatenatedData=numpy.empty(0)
	
		for oneSessionKind in sessionKinds:
		
			thisDict=[element for element in oneDictList if element['sessionKind']==oneSessionKind][0]
		
			concatenatedData=numpy.append(concatenatedData,thisDict[timeSeriesString])
		
			for oneRegressorString in regressorStrings:
			
				if oneRegressorString in thisDict.keys():
				
					if 'probeReports' in oneRegressorString and 'Active' in oneSessionKind:	#some people pressed space a few times when there weren't actually any probes
						concatenatedDict[oneRegressorString+'_'+oneSessionKind]=numpy.array([])
					else:
						concatenatedDict[oneRegressorString+'_'+oneSessionKind]=thisDict[oneRegressorString]+totalOffset
						
			for oneShiftString in shiftStrings:
			
				if oneShiftString in thisDict.keys():
				
					concatenatedDict[oneShiftString]=thisDict[oneShiftString]

			trialStartMoments=thisDict['trialStarts']
			for oneCovariateListPre in covariateListsPre:
			
				if 'probeReports' in oneCovariateListPre[0] and 'Active' in oneSessionKind:	#some people pressed space a few times when there weren't actually any probes
				
					concatenatedDict[oneCovariateListPre[2]+'_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListPre[2]+'NoZscore_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListPre[2]+'ZscorePerPercept_'+oneSessionKind]=numpy.array([])
					
				else:
					
					theRegressorMoments=thisDict[oneCovariateListPre[0]]
					thePrecedingMoments=thisDict[oneCovariateListPre[1][0]]		#oneCovariateListPre[1] is a list of several (or just 1) keywords: the last moment out of all those event types is chosen
					
					for anyOtherListMember in oneCovariateListPre[1][1:]:
						thePrecedingMoments=numpy.append(thePrecedingMoments,thisDict[anyOtherListMember])

					thePrecedingMoments.sort()	
			
					intervalValues=[]
					intervalValuesLog=[]
					noIntervalIndices=[]
					for oneRegressorIndex,oneRegressorMoment in enumerate(theRegressorMoments):
				
						precedingReferenceMoments=[element for element in thePrecedingMoments if element<oneRegressorMoment]
						if len(precedingReferenceMoments)>0:
							precedingReferenceMoment=precedingReferenceMoments[-1]
							anyTrialStartsBetween=[element for element in trialStartMoments if element<oneRegressorMoment and element>precedingReferenceMoment]
							if len(anyTrialStartsBetween)>0:
								intervalValues=intervalValues+['n/a']
								intervalValuesLog=intervalValuesLog+['n/a']
								noIntervalIndices=noIntervalIndices+[oneRegressorIndex]
							else:
								diff=oneRegressorMoment-precedingReferenceMoment
								intervalValues=intervalValues+[diff]
								intervalValuesLog=intervalValuesLog+[numpy.log(diff)]	#log of interval taken
						else:
							intervalValues=intervalValues+['n/a']
							intervalValuesLog=intervalValuesLog+['n/a']
							noIntervalIndices=noIntervalIndices+[oneRegressorIndex]
			
					validValues=[intervalValues[thisIndex] for thisIndex in range(len(intervalValues)) if not thisIndex in noIntervalIndices]
					validValuesLog=[intervalValuesLog[thisIndex] for thisIndex in range(len(intervalValuesLog)) if not thisIndex in noIntervalIndices]
					
					theMean=numpy.average(validValuesLog)
					theStd=numpy.std(validValuesLog)
					
					if oneCovariateListPre[0][:-2]+'Identities' in thisDict.keys():
						theSwitchIdentities=thisDict[oneCovariateListPre[0][:-2]+'Identities']
						theMeanPerIdentity=[]
						theStdPerIdentity=[]
						for identity in [-1.,1.]:
							validValuesLogThisIdentity=[intervalValuesLog[thisIndex] for thisIndex in range(len(intervalValuesLog)) if theSwitchIdentities[thisIndex]==identity and not intervalValuesLog[thisIndex]=='n/a']
							
							theMeanPerIdentity=theMeanPerIdentity+[numpy.average(validValuesLogThisIdentity)]
							theStdPerIdentity=theStdPerIdentity+[numpy.std(validValuesLogThisIdentity)]
						
						zScoredPerIdentity=[]
						for oneIntervalIndex,oneIntervalValueLog in enumerate(intervalValuesLog):
							if oneIntervalValueLog=='n/a':
								zScoredPerIdentity=zScoredPerIdentity+[0]
							else:
								theIdentityIndex=int((theSwitchIdentities[oneIntervalIndex]+1.)/2.)		#-1. and -1. become 0 and 1
								zScoredPerIdentity=zScoredPerIdentity+[(oneIntervalValueLog-theMeanPerIdentity[theIdentityIndex])/theStdPerIdentity[theIdentityIndex]]
					else:
						zScoredPerIdentity='n/a'
						
					for thisIndex in noIntervalIndices:
						intervalValuesLog[thisIndex]=theMean #if no interval available, give it the mean value (which will become 0 after z-scoring)
					intervalValuesZscored=(numpy.array(intervalValuesLog)-theMean)/theStd
					
					concatenatedDict[oneCovariateListPre[2]+'_'+oneSessionKind]=intervalValuesZscored		#not just z-scored: also first log()
					concatenatedDict[oneCovariateListPre[2]+'NoZscore_'+oneSessionKind]=validValues		#no z-score; no log() either
					concatenatedDict[oneCovariateListPre[2]+'ZscorePerPercept_'+oneSessionKind]=zScoredPerIdentity	#not just z-scored: also first log()	
					
			trialEndMoments=thisDict['trialEnds']
		
			for oneCovariateListPost in covariateListsPost:
			
				if 'probeReports' in oneCovariateListPost[0] and 'Active' in oneSessionKind:	#some people pressed space a few times when there weren't actually any probes
				
					concatenatedDict[oneCovariateListPost[2]+'_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListPost[2]+'NoZscore_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListPost[2]+'ZscorePerPercept_'+oneSessionKind]=numpy.array([])
					
				else:

					theRegressorMoments=thisDict[oneCovariateListPost[0]]
					theFollowingMoments=thisDict[oneCovariateListPost[1][0]]		#oneCovariateListPost[1] is a list of several (or just 1) keywords: the first moment out of all those event types is chosen

					for anyOtherListMember in oneCovariateListPost[1][1:]:
						theFollowingMoments=numpy.append(theFollowingMoments,thisDict[anyOtherListMember])
				
					theFollowingMoments.sort()

					intervalValues=[]
					intervalValuesLog=[]
					noIntervalIndices=[]
					for oneRegressorIndex,oneRegressorMoment in enumerate(theRegressorMoments):

						followingReferenceMoments=[element for element in theFollowingMoments if element>oneRegressorMoment]
						if len(followingReferenceMoments)>0:
							followingReferenceMoment=followingReferenceMoments[0]
							anyTrialEndsBetween=[element for element in trialEndMoments if element>oneRegressorMoment and element<followingReferenceMoment]
							if len(anyTrialEndsBetween)>0:
								intervalValues=intervalValues+['n/a']
								intervalValuesLog=intervalValuesLog+['n/a']
								noIntervalIndices=noIntervalIndices+[oneRegressorIndex]
							else:
								diff=followingReferenceMoment-oneRegressorMoment
								intervalValues=intervalValues+[diff]
								intervalValuesLog=intervalValuesLog+[numpy.log(diff)]
						else:
							intervalValues=intervalValues+['n/a']
							intervalValuesLog=intervalValuesLog+['n/a']
							noIntervalIndices=noIntervalIndices+[oneRegressorIndex]

					validValues=[intervalValues[thisIndex] for thisIndex in range(len(intervalValues)) if not thisIndex in noIntervalIndices]
					validValuesLog=[intervalValuesLog[thisIndex] for thisIndex in range(len(intervalValuesLog)) if not thisIndex in noIntervalIndices]
					
					theMean=numpy.average(validValuesLog)
					theStd=numpy.std(validValuesLog)

					if oneCovariateListPost[0][:-2]+'Identities' in thisDict.keys():
						theSwitchIdentities=thisDict[oneCovariateListPost[0][:-2]+'Identities']
						theMeanPerIdentity=[]
						theStdPerIdentity=[]
						for identity in [-1.,1.]:
							validValuesLogThisIdentity=[intervalValuesLog[thisIndex] for thisIndex in range(len(intervalValuesLog)) if theSwitchIdentities[thisIndex]==identity and not intervalValuesLog[thisIndex]=='n/a']
							
							theMeanPerIdentity=theMeanPerIdentity+[numpy.average(validValuesLogThisIdentity)]
							theStdPerIdentity=theStdPerIdentity+[numpy.std(validValuesLogThisIdentity)]
						
						zScoredPerIdentity=[]
						for oneIntervalIndex,oneIntervalValueLog in enumerate(intervalValuesLog):
							if oneIntervalValueLog=='n/a':
								zScoredPerIdentity=zScoredPerIdentity+[0]
							else:
								theIdentityIndex=int((theSwitchIdentities[oneIntervalIndex]+1.)/2.)		#-1. and -1. become 0 and 1
								zScoredPerIdentity=zScoredPerIdentity+[(oneIntervalValueLog-theMeanPerIdentity[theIdentityIndex])/theStdPerIdentity[theIdentityIndex]]
					else:
						zScoredPerIdentity='n/a'
									
					for thisIndex in noIntervalIndices:
						intervalValuesLog[thisIndex]=theMean #if no interval available, give it the mean value (which will become 0 after z-scoring)
					intervalValuesZscored=(numpy.array(intervalValuesLog)-theMean)/theStd

					concatenatedDict[oneCovariateListPost[2]+'_'+oneSessionKind]=intervalValuesZscored		#not just z-scored: also first log()
					concatenatedDict[oneCovariateListPost[2]+'NoZscore_'+oneSessionKind]=validValues		#no z-score; no log()
					concatenatedDict[oneCovariateListPost[2]+'ZscorePerPercept_'+oneSessionKind]=zScoredPerIdentity	#not just z-scored: also first log()	
		
			for oneCovariateListOther in covariateListsOther:
		
				if 'probeReports' in oneCovariateListOther[0] and 'Active' in oneSessionKind:	#some people pressed space a few times when there weren't actually any probes
			
					concatenatedDict[oneCovariateListOther[2]+'_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListOther[2]+'NoZscore_'+oneSessionKind]=numpy.array([])
					concatenatedDict[oneCovariateListOther[2]+'ZscorePerPercept_'+oneSessionKind]=numpy.array([])
				
				else:

					theRegressorMoments=thisDict[oneCovariateListOther[0]]
					theCovariateValues=[element for element in thisDict[oneCovariateListOther[1]]]
					theCovariateValuesLog=[numpy.log(element) for element in thisDict[oneCovariateListOther[1]]]

					theMean=numpy.average(theCovariateValuesLog)
					theStd=numpy.std(theCovariateValuesLog)

					if oneCovariateListOther[0][:-2]+'Identities' in thisDict.keys():
						theSwitchIdentities=thisDict[oneCovariateListOther[0][:-2]+'Identities']
						theMeanPerIdentity=[]
						theStdPerIdentity=[]
						for identity in [-1.,1.]:
							covariateValuesThisIdentity=[theCovariateValuesLog[thisIndex] for thisIndex in range(len(theCovariateValuesLog)) if theSwitchIdentities[thisIndex]==identity]
							theMeanPerIdentity=theMeanPerIdentity+[numpy.average(covariateValuesThisIdentity)]
							theStdPerIdentity=theStdPerIdentity+[numpy.std(covariateValuesThisIdentity)]
					
						zScoredPerIdentity=[]
						for oneCovariateIndex,oneCovariateValue in enumerate(theCovariateValuesLog):
							theIdentityIndex=int((theSwitchIdentities[oneCovariateIndex]+1.)/2.)		#-1. and -1. become 0 and 1
							zScoredPerIdentity=zScoredPerIdentity+[(oneCovariateValue-theMeanPerIdentity[theIdentityIndex])/theStdPerIdentity[theIdentityIndex]]
							
					else:
						zScoredPerIdentity='n/a'
								
					covariateValuesZscored=(numpy.array(theCovariateValuesLog)-theMean)/theStd

					concatenatedDict[oneCovariateListOther[2]+'_'+oneSessionKind]=covariateValuesZscored		#not just z-scored: also first log()
					concatenatedDict[oneCovariateListOther[2]+'NoZscore_'+oneSessionKind]=theCovariateValues		#no z-score; no log()
					concatenatedDict[oneCovariateListOther[2]+'ZscorePerPercept_'+oneSessionKind]=zScoredPerIdentity	#not just z-scored: also first log()	
		
			for oneSplitList in splitLists:

				thisRegressor=oneSplitList[0]

				for oneKey in concatenatedDict.keys():

					if thisRegressor+'_' in oneKey:

						accompanyingCovariate=oneSplitList[1]+'_'+oneKey.split('_')[1]

						covariateValues=concatenatedDict[accompanyingCovariate]
						
						if len(covariateValues)>oneSplitList[2]:
							
							covariateValuesForSorting=copy.deepcopy(covariateValues)
							covariateValuesForSorting.sort()
							
							cutOffs=[]
							for numerator in range(oneSplitList[2]):
								proportion=float(numerator)/float(oneSplitList[2])
								cutOffs=cutOffs+[covariateValuesForSorting[int(proportion*len(covariateValuesForSorting))]]

							cutOffs=cutOffs+[numpy.inf]

							regressorValues=concatenatedDict[oneKey]
							for sectionIndex in range(oneSplitList[2]):
								regressorValsThisSection=[regressorValues[thisIndex] for thisIndex in range(len(regressorValues)) if covariateValues[thisIndex]>=cutOffs[sectionIndex] and covariateValues[thisIndex]<cutOffs[sectionIndex+1]]
								concatenatedDict[oneKey.split('_')[0]+'SPLITBY'+oneSplitList[1]+str(sectionIndex)+'_'+oneKey.split('_')[1]]=numpy.array(regressorValsThisSection)
							
						else:
							
							for sectionIndex in range(oneSplitList[2]):
								concatenatedDict[oneKey.split('_')[0]+'SPLITBY'+oneSplitList[1]+str(sectionIndex)+'_'+oneKey.split('_')[1]]=numpy.array([])
									
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
		
		for oneRegressorString in regressorStrings:	#join regressors of a given kind across all conditions (i.e. all like-named ones).
		
			jointRegressorAcrossConditions=numpy.empty(0)
		
			for oneSessionKind in sessionKinds:
			
				if oneRegressorString+'_'+oneSessionKind in concatenatedDict.keys():
				
					jointRegressorAcrossConditions=numpy.append(jointRegressorAcrossConditions,concatenatedDict[oneRegressorString+'_'+oneSessionKind])
				
			#jointRegressorAcrossConditions.sort() #don't sort, because if there's an accompanying covariate then you want to keep them matched up. Actually, I don't think this remark makes any sense because the covariates aren't joined in this way. But that's fine: sorting is inessential.
			concatenatedDict[oneRegressorString+'_Any']=jointRegressorAcrossConditions
			
			#also join any sectioned-up versions of this regressor, if they exist. This order of doing things means that here we're splitting by the covariates within conditions, and only then pooling across conditions; not pooling the individual covariate values across conditions.
			for oneSplitList in splitLists:
				
				if oneSplitList[0]==oneRegressorString:
					
					splitByString=oneSplitList[1]
					numSections=oneSplitList[2]
			
					for oneSectionIndex in range(numSections):
						
						jointRegressorAcrossConditions=numpy.empty(0)
						
						splitKeyNoSessionType=oneRegressorString+'SPLITBY'+splitByString+str(oneSectionIndex)
			
						for oneSessionKind in sessionKinds:
							
							splitKey=splitKeyNoSessionType+'_'+oneSessionKind
							
							if splitKey in concatenatedDict.keys():
								
								jointRegressorAcrossConditions=numpy.append(jointRegressorAcrossConditions,concatenatedDict[splitKey])
								
						concatenatedDict[splitKeyNoSessionType+'_Any']=jointRegressorAcrossConditions
								
		
		for oneCovariateList in covariateListsPre+covariateListsPost+covariateListsOther:	#automatically join covariates of a given kind across all conditions. This will produce an ordering that matches the one produced by the regressor joining right above this

			jointCovariateAcrossConditions=numpy.empty(0)
			jointCovariateAcrossConditionsNoZscore=numpy.empty(0)
			jointCovariateAcrossConditionsZscorePerPercept=numpy.empty(0)

			for oneSessionKind in sessionKinds:

				if oneCovariateList[2]+'_'+oneSessionKind in concatenatedDict.keys():

					jointCovariateAcrossConditions=numpy.append(jointCovariateAcrossConditions,concatenatedDict[oneCovariateList[2]+'_'+oneSessionKind])
					jointCovariateAcrossConditionsNoZscore=numpy.append(jointCovariateAcrossConditionsNoZscore,concatenatedDict[oneCovariateList[2]+'NoZscore_'+oneSessionKind])
					jointCovariateAcrossConditionsZscorePerPercept=numpy.append(jointCovariateAcrossConditionsZscorePerPercept,concatenatedDict[oneCovariateList[2]+'ZscorePerPercept_'+oneSessionKind])

			concatenatedDict[oneCovariateList[2]+'_Any']=jointCovariateAcrossConditions
			concatenatedDict[oneCovariateList[2]+'NoZscore_Any']=jointCovariateAcrossConditionsNoZscore
			concatenatedDict[oneCovariateList[2]+'ZscorePerPercept_Any']=jointCovariateAcrossConditionsZscorePerPercept
			
		for oneJoinSet in joinSets:	#join regressors of various kinds into a single regressor (e.g. all key presses)
		
			sourceData=numpy.empty(0)
			for oneSourceRegressor in oneJoinSet[0]:
				sourceData=numpy.append(sourceData,concatenatedDict[oneSourceRegressor])
				
			#sourceData.sort()	#don't sort, because if there's an accompanying covariate then you want to keep them matched up
			concatenatedDict[oneJoinSet[1]]=sourceData
			
		for oneJoinSetCovariates in joinSetsCovariates:	
			
			#------merge the covariates together
			sourceData=numpy.empty(0)
			sourceDataNoZscore=numpy.empty(0)
			sourceDataZscorePerPercept=numpy.empty(0)
			for oneSessionType in oneJoinSetCovariates[1]:
				sourceData=numpy.append(sourceData,concatenatedDict[oneJoinSetCovariates[0]+'_'+oneSessionType])
				sourceDataNoZscore=numpy.append(sourceDataNoZscore,concatenatedDict[oneJoinSetCovariates[0]+'NoZscore_'+oneSessionType])
				sourceDataZscorePerPercept=numpy.append(sourceDataZscorePerPercept,concatenatedDict[oneJoinSetCovariates[0]+'ZscorePerPercept_'+oneSessionType])
			
			concatenatedDict[oneJoinSetCovariates[0]+'_'+oneJoinSetCovariates[2]]=sourceData
			concatenatedDict[oneJoinSetCovariates[0]+'NoZscore_'+oneJoinSetCovariates[2]]=sourceDataNoZscore
			concatenatedDict[oneJoinSetCovariates[0]+'ZscorePerPercept_'+oneJoinSetCovariates[2]]=sourceDataNoZscore			
			#------------------------
			#------but also merge the events that the covariates belong to together
			#------and also create a regressor to capture the difference between the two session types being combined
			
			if not len(oneJoinSetCovariates[1])==2:
				
				print('Do not know how to deal with difference between more than 2 session types.')	#because you're using one of those contrast regressors here to distinguish two conditions while still applying a common covariate to both.
				shell()
				
			else:
					
				thisEventType=[element for element in covariateListsPre+covariateListsPost+covariateListsOther if element[-1]==oneJoinSetCovariates[0]][0][0]
				
				sessionTypeDiffRegressor=numpy.empty(0)
				sourceData=numpy.empty(0)
				
				multipliers=[-1.,1.]
				
				for oneSessionIndex,oneSessionType in enumerate(oneJoinSetCovariates[1]):
					sourceData=numpy.append(sourceData,concatenatedDict[thisEventType+'_'+oneSessionType])
					
					numObs=len(concatenatedDict[thisEventType+'_'+oneSessionType])
					regressorVal=multipliers[oneSessionIndex]/float(numObs)
					sessionTypeDiffRegressor=numpy.append(sessionTypeDiffRegressor,numpy.ones(numObs)*regressorVal)
				
				concatenatedDict[thisEventType+'_'+oneJoinSetCovariates[2]]=sourceData
				concatenatedDict[thisEventType+'_'+oneJoinSetCovariates[2]+'_contrastRegressor']=sessionTypeDiffRegressor
			
			#-----------------------
		
		for oneJoinSetSplitList in joinSetsSplitLists:	
			
			#------merge the splitLists together
			for oneSectionIndex in range(numSections):
				
				sourceData=numpy.empty(0)
				for oneSessionType in oneJoinSetSplitList[1]:
					sourceData=numpy.append(sourceData,concatenatedDict[oneJoinSetSplitList[0]+str(oneSectionIndex)+'_'+oneSessionType])
			
				concatenatedDict[oneJoinSetSplitList[0]+str(oneSectionIndex)+'_'+oneJoinSetSplitList[2]]=sourceData		

			#-----------------------
		del oneDictList	#clear up memory
		
		with open(myPath+miscStuffSubFolder+'/'+oneObs+'_concatenatedDictList', 'w') as f:
		    pickle.dump(concatenatedDict, f)
	
	concatenatedDictList=concatenatedDictList+[concatenatedDict]

#----------------
#define GLMS
print('Watch out. By default z-scoring intervals surrounding probe reports (or other covariates) per condition; not across. Z-scoring per percept is available.')
print('Watch out #2: currently automatically taking the log of covariates befor z-scoring. Which makes sense if we\'re talking about time intervals.')
print('None of this is about pupil size z-scoring: this is all about covariate z-scoring')

plotDictList=[]

#for plots, some commented out:
#
# plotTitle='_Concatenated_active_rivalry_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Passive rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Passive rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_split_pre'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPre'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_split_pre'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_split_post'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPost'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_split_post'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPost'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_split_rivalry_pre_anySwitches'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Any rivalry' for splitIndex in range(numSections)]+['keyPress_Any','inferredSwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_split_rivalry_post_anySwitches'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Any rivalry' for splitIndex in range(numSections)]+['keyPress_Any','inferredSwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_anySwitches_split_any_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Any' for splitIndex in range(numSections)]+['keyPress_Any','inferredRivalrySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_anySwitches_split_any_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Any' for splitIndex in range(numSections)]+['keyPress_Any','inferredRivalrySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# #figs 4 and 5
# #-----
# plotTitle='_Concatenated_keys_rivalrySwitches_split_rivalry_pre_replaySwitches'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Any rivalry' for splitIndex in range(numSections)]+['keyPress_Any','inferredReplaySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_split_rivalry_post_replaySwitches'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Any rivalry' for splitIndex in range(numSections)]+['keyPress_Any','inferredReplaySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_replaySwitches_split_replay_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Any replay' for splitIndex in range(numSections)]+['keyPress_Any','inferredRivalrySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_replaySwitches_split_replay_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Any replay' for splitIndex in range(numSections)]+['keyPress_Any','inferredRivalrySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['key','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
# #-----
#
# plotTitle='_Concatenated_passive_replay_inferred0duration_split_pre'
# regressors=[['inferredSwitches0durationSPLITBYcovInfSwitch0durInfIntervalPre'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitchesNon0duration_Passive replay','inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported0duration_split_post'
# regressors=[['reportedSwitches0durationSPLITBYcovRepSwitch0durRepIntervalPost'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['reportedSwitchesNon0duration_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferredNoProbe_split_post'
# regressors=[['inferredSwitchesNoProbeInSubsequentPercSPLITBYcovInfSwitchNoProbeSubsqInfIntervalPost'+str(splitIndex)+'_Passive rivalry' for splitIndex in range(numSections)]+['inferredSwitchesProbeInSubsequentPerc_Passive rivalry','inferredSwitches_Active replay','inferredSwitches_Active rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

#key-based subsequent percept analysis for comparison to Einhauser en Hollander
plotTitle='_Concatenated_active_rivalry_reported_split_post'
regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPost'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
plottedRegressorIndices=[range(numSections)]
timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
alignmentInfo=[['key' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
calculate_var_explained=False
plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

plotTitle='_Concatenated_active_replay_reported_split_post'
regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPost'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
plottedRegressorIndices=[range(numSections)]
timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
alignmentInfo=[['key' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
calculate_var_explained=False
plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]


#and for stats, some commented out:
#
# plotTitle='_Concatenated_active_rivalry_inferred_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Active rivalry','covInfSwitchInfIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Active replay','covInfSwitchInfIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covInfSwitchInfIntervalPre_Passive rivalry','covInfSwitchInfIntervalPost_Passive rivalry'],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covInfSwitchInfIntervalPre_Passive replay','covInfSwitchInfIntervalPost_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_covariates_switchmoment_derivative'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepIntervalPre_Active rivalry','covRepSwitchRepIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_covariates_switchmoment_derivative'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covRepSwitchRepIntervalPre_Active replay','covRepSwitchRepIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_keys_rivalrySwitches_covariates_switchmoment_anySwitches_covariates_switchmoment_derivative'
# regressors=[['keyPress_Any','inferredRivalrySwitches_Any','inferredSwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Any rivalry','covInfSwitchInfIntervalPost_Any rivalry'],['covInfSwitchInfIntervalPre_Any','covInfSwitchInfIntervalPost_Any'],[],[],[],[]]]
# plottedRegressorIndices=[[1,2]]
# timeShifts=[[0,0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# #figures 4 and 5
# #-------
# plotTitle='_Concatenated_keys_rivalrySwitches_covariates_switchmoment_replaySwitches_covariates_switchmoment_derivative'
# regressors=[['keyPress_Any','inferredRivalrySwitches_Any','inferredReplaySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any','unreportedProbes_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Any rivalry','covInfSwitchInfIntervalPost_Any rivalry'],['covInfSwitchInfIntervalPre_Any replay','covInfSwitchInfIntervalPost_Any replay'],[],[],[],[]]]
# plottedRegressorIndices=[[1,2]]
# timeShifts=[[0,0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
# #-------
#
# plotTitle='_Concatenated_passive_replay_inferred0duration_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches0duration_Passive replay','inferredSwitchesNon0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covInfSwitch0durInfIntervalPre_Passive replay','covInfSwitch0durInfIntervalPost_Passive replay'],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported0duration_covariates_switchmoment_derivative'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches0duration_Active rivalry','reportedSwitchesNon0duration_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitch0durRepIntervalPre_Active rivalry','covRepSwitch0durRepIntervalPost_Active rivalry'],[],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferredNoProbeSubsq_covariates_switchmoment_derivative'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['inferredSwitchesNoProbeInSubsequentPerc_Passive rivalry','inferredSwitchesProbeInSubsequentPerc_Passive rivalry','inferredSwitches_Active replay','inferredSwitches_Active rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchNoProbeSubsqInfIntervalPre_Passive rivalry','covInfSwitchNoProbeSubsqInfIntervalPost_Passive rivalry'],[],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# #figure z
# #--------
#
# plotTitle='_Concatenated_passive_rivalry_inferred_noblinks_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitchesNoBlink_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','inferredSwitchesBlink_Passive rivalry']]
# covariates=[[[],[],['covInfSwitchNoBlinkInfIntervalPre_Passive rivalry','covInfSwitchNoBlinkInfIntervalPost_Passive rivalry'],
# [],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz','OKN']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_noblinks_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitchesNoBlink_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','inferredSwitchesBlink_Passive replay']]
# covariates=[[[],[],[],
# ['covInfSwitchNoBlinkInfIntervalPre_Passive replay','covInfSwitchNoBlinkInfIntervalPost_Passive replay'],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz','OKN']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

#figure AA
#--------
#
# plotTitle='_Concatenated_any_passive_probereport_covariates_probereport_derivative'
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# [],[],['covProbeRepProbeRepIntervalPre_Any','covProbeRepProbeRepIntervalPost_Any'],[],[],[]]]
# plottedRegressorIndices=[[5]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

# #Pupil area itself, to verify whether any subsequent effect present when using that measure
# plotTitle='_Concatenated_active_rivalry_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Active rivalry','covInfSwitchInfIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Active replay','covInfSwitchInfIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covInfSwitchInfIntervalPre_Passive rivalry','covInfSwitchInfIntervalPost_Passive rivalry'],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covInfSwitchInfIntervalPre_Passive replay','covInfSwitchInfIntervalPost_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

#key-based subsequent percept analysis for comparison to Einhauser en Hollander

plotTitle='_Concatenated_active_rivalry_reported_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
covariates=[[['covRepSwitchRepIntervalPre_Active rivalry','covRepSwitchRepIntervalPost_Active rivalry'],[],[],
[],[],[],[],[],[]]]
plottedRegressorIndices=[[0,1,2,3]]
timeShifts=[[0,0,0,0,0,0,0,0,0]]
alignmentInfo=[['key','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
calculate_var_explained=False
plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]

plotTitle='_Concatenated_active_replay_reported_covariates_switchmoment_derivative'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
regressors=[['inferredSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
covariates=[[[],['covRepSwitchRepIntervalPre_Active replay','covRepSwitchRepIntervalPost_Active replay'],[],
[],[],[],[],[],[]]]
plottedRegressorIndices=[[0,1,2,3]]
timeShifts=[[0,0,0,0,0,0,0,0,0]]
alignmentInfo=[['OKN','key','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
calculate_var_explained=False
plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]









#---------------------------------------
#obsolete:
# plotTitle='_Concatenated_keys_rivalrySwitches_replayedSwitchesInferred_covariates_switchmoment_derivative_zscore'
# regressors=[['keyPress_Any','inferredRivalrySwitch_Any','inferredReplaySwitches_Any','saccades_Any','blinks_Any','trialStarts_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Any rivalry','covInfSwitchInfIntervalPost_Any rivalry'],['covInfSwitchInfIntervalPre_Any replay','covInfSwitchInfIntervalPost_Any replay'],[],[]]]
# plottedRegressorIndices=[[1,2]]
# timeShifts=[[0,0,0,0,0,0]]
# alignmentInfo=[['key','OKN','OKN','n/a','n/a','n/a']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]




# other options not really used anymore (some duplicates):
#
# plotTitle='_Concatenated_active_rivalry_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Active rivalry','covInfSwitchInfIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPre_Active replay','covInfSwitchInfIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covInfSwitchInfIntervalPre_Passive rivalry','covInfSwitchInfIntervalPost_Passive rivalry'],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covInfSwitchInfIntervalPre_Passive replay','covInfSwitchInfIntervalPost_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# # plotTitle='_Concatenated_rivalry_vs_replay_inferred_covariates_switchmoment_complete'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# # regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# # covariates=[[['covInfSwitchInfIntervalPre_Active rivalry','covInfSwitchInfIntervalPost_Active rivalry'],['covInfSwitchInfIntervalPre_Active replay','covInfSwitchInfIntervalPost_Active replay'],['covInfSwitchInfIntervalPre_Passive rivalry','covInfSwitchInfIntervalPost_Passive rivalry'],
# # ['covInfSwitchInfIntervalPre_Passive replay','covInfSwitchInfIntervalPost_Passive replay'],[],['covProbeReportReportIntervalPre_Any','covProbeReportReportIntervalPost_Any'],[],[],[]]]
# # plottedRegressorIndices=[[0,1,2,3]]
# # timeShifts=[[0,0,0,0,0,0,0,0,0]]
# # alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# # calculate_var_explained=False
# # plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
# #
# plotTitle='_Concatenated_probe_reports'
# regressors=[['reportedSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# [],[],['covProbeReportReportIntervalPre_Any','covProbeReportReportIntervalPost_Any'],[],[],[]]]
# plottedRegressorIndices=[[5]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','tevz','tevz','n/a','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_physical_covariates_switchmoment'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covPhysSwitchPhysIntervalPre_Active replay','covPhysSwitchPhysIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_physical_covariates_switchmoment'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covPhysSwitchPhysIntervalPre_Passive replay','covPhysSwitchPhysIntervalPost_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_physical_covariates_switchmoment'    #use only instaneous switches as regressors, but for times relative to those switches, use whatever the neighboring switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],['covPhysSwitch0durPhysIntervalPre_Active replay','covPhysSwitch0durPhysIntervalPost_Active replay'],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_instantaneous_physical_covariates_switchmoment'    #use only instaneous switches as regressors, but for times relative to those switches, use whatever the neighboring switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],[],[],
# ['covPhysSwitchPhysIntervalPre_Passive replay','covPhysSwitchPhysIntervalPost_Passive replay'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_physical_covariates_instantaneousstartend'#use only instaneous switches as regressors, and for times relative to those switches, use the start of the next switch / end of previous one, if those are not instantaneous
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],['covPhysSwitch0durPhysEndIntervalPre_Active replay','covPhysSwitch0durPhysStartIntervalPost_Active replay'],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_instantaneous_physical_covariates_instantaneousstartend'#use only instaneous switches as regressors, and for times relative to those switches, use the start of the next switch / end of previous one, if those are not instantaneous
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches0duration_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],[],[],
# ['covPhysSwitch0durPhysEndIntervalPre_Passive replay','covPhysSwitch0durPhysStartIntervalPost_Passive replay'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_covariates_switchmoment'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepIntervalPre_Active rivalry','covRepSwitchRepIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_covariates_switchmoment'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covRepSwitchRepIntervalPre_Active replay','covRepSwitchRepIntervalPost_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_instantaneous_reported_covariates_switchmoment'    #use only instaneous switches as regressors, but for times relative to those switches, use whatever the neighboring switch is marked as (can be halfway a true transition)
# regressors=[['reportedSwitches0duration_Active rivalry','reportedSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[['covRepSwitch0durRepIntervalPre_Active rivalry','covRepSwitch0durRepIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_reported_covariates_switchmoment'    #use only instaneous switches as regressors, but for times relative to those switches, use whatever the neighboring switch is marked as (can be halfway a true transition)
# regressors=[['reportedSwitches0duration_Active rivalry','reportedSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],['covRepSwitch0durRepIntervalPre_Active replay','covRepSwitch0durRepIntervalPost_Active replay'],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_instantaneous_reported_covariates_instantaneousstartend'#use only instaneous switches as regressors, and for times relative to those switches, use the start of the next switch / end of previous one, if those are not instantaneous
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[['covRepSwitch0durRepEndIntervalPre_Active rivalry','covRepSwitch0durRepStartIntervalPost_Active rivalry'],[],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_reported_covariates_instantaneousstartend'#use only instaneous switches as regressors, and for times relative to those switches, use the start of the next switch / end of previous one, if those are not instantaneous
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches0duration_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],['covRepSwitch0durRepEndIntervalPre_Active replay','covRepSwitch0durRepStartIntervalPost_Active replay'],[],
# [],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_joint_inferred_covariates_switchmoment'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Any','keyPress_Any','trialStarts_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Any','covInfSwitchInfIntervalPost_Any'],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_rivalry_inferred_covariates_switchmoment'
# regressors=[['inferredSwitches_Any rivalry','physicalSwitches_Active replay','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Any rivalry','covInfSwitchInfIntervalPost_Any rivalry','inferredSwitches_Any rivalry_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_inferred_covariates_switchmoment'
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covInfSwitchInfIntervalPre_Any replay','covInfSwitchInfIntervalPost_Any replay','inferredSwitches_Any replay_contrastRegressor'],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_physical_covariates_switchmoment'
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','physicalSwitches_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covPhysSwitchPhysIntervalPre_Any replay','covPhysSwitchPhysIntervalPost_Any replay','physicalSwitches_Any replay_contrastRegressor'],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_instantaneous_physical_covariates_switchmoment'    #use only instaneous switches as regressors, but for times relative to those switches, use whatever the neighboring switch is marked as (can be halfway a true transition)
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','physicalSwitches0duration_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],[],['covPhysSwitch0durPhysIntervalPre_Any replay','covPhysSwitch0durPhysIntervalPost_Any replay','physicalSwitches0duration_Any replay_contrastRegressor'],
# [],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_instantaneous_physical_covariates_instantaneousstartend'#use only instaneous switches as regressors, and for times relative to those switches, use the start of the next switch / end of previous one, if those are not instantaneous
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','physicalSwitches0duration_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','physicalSwitchesNon0duration_Active replay','physicalSwitchesNon0duration_Passive replay']]
# covariates=[[[],[],['covPhysSwitch0durPhysEndIntervalPre_Any replay','covPhysSwitch0durPhysStartIntervalPost_Any replay','physicalSwitches0duration_Any replay_contrastRegressor'],
# [],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_active_inferred_covariates_switchmoment'
# regressors=[['inferredSwitches_Any active','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Any active','covInfSwitchInfIntervalPost_Any active','inferredSwitches_Any active_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_passive_inferred_covariates_switchmoment'
# regressors=[['inferredSwitches_Any passive','reportedSwitches_Active rivalry','physicalSwitches_Active replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPre_Any passive','covInfSwitchInfIntervalPost_Any passive','inferredSwitches_Any passive_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_active_reported_covariates_switchmoment'
# regressors=[['reportedSwitches_Any active','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepIntervalPre_Any active','covRepSwitchRepIntervalPost_Any active','reportedSwitches_Any active_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_inferred_covariates_switchmoment_withinPerceptZ'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPreZscorePerPercept_Active rivalry','covInfSwitchInfIntervalPostZscorePerPercept_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_covariates_switchmoment_withinPerceptZ'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covInfSwitchInfIntervalPreZscorePerPercept_Active replay','covInfSwitchInfIntervalPostZscorePerPercept_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_covariates_switchmoment_withinPerceptZ'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covInfSwitchInfIntervalPreZscorePerPercept_Passive rivalry','covInfSwitchInfIntervalPostZscorePerPercept_Passive rivalry'],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_covariates_switchmoment_withinPerceptZ'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],
# ['covInfSwitchInfIntervalPreZscorePerPercept_Passive replay','covInfSwitchInfIntervalPostZscorePerPercept_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[0,1,2,3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN','OKN','OKN','OKN','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_joint_inferred_covariates_switchmoment_withinPerceptZ'        #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition)
# regressors=[['inferredSwitches_Any','keyPress_Any','trialStarts_Any','saccades_Any','blinks_Any']]
# covariates=[[['covInfSwitchInfIntervalPreZscorePerPercept_Any','covInfSwitchInfIntervalPostZscorePerPercept_Any'],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0]]
# alignmentInfo=[['OKN','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_active_instantaneous_reported_covariates_switchmoment'
# regressors=[['reportedSwitches0duration_Any active','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any','reportedSwitchesNon0duration_Passive rivalry','reportedSwitchesNon0duration_Passive replay']]
# covariates=[[['covRepSwitch0durRepIntervalPre_Any active','covRepSwitch0durRepIntervalPost_Any active','reportedSwitches0duration_Any active_contrastRegressor'],[],[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# #--------
# #split pre
#
# plotTitle='_Concatenated_active_rivalry_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Passive rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_joint_inferred_split_pre'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPre'+str(splitIndex)+'_Any' for splitIndex in range(numSections)]+['keyPress_Any','trialStarts_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
#
# #split pre
#
# plotTitle='_Concatenated_active_rivalry_reported_split_pre'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPre'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_split_pre'
# regressors=[['reportedSwitchesSPLITBYcovRepSwitchRepIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_instantaneous_reported_split_pre'
# regressors=[['reportedSwitches0durationSPLITBYcovRepSwitch0durRepIntervalPre'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_reported_split_pre'
# regressors=[['reportedSwitches0durationSPLITBYcovRepSwitch0durRepIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_instantaneous_physical_split_pre'
# regressors=[['physicalSwitches0durationSPLITBYcovPhysSwitch0durPhysIntervalPre'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Passive rivalry','inferredSwitches_Active rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_instantaneous_physical_split_pre'
# regressors=[['physicalSwitches0durationSPLITBYcovPhysSwitch0durPhysIntervalPre'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Active replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# #-----
# #post
#
# plotTitle='_Concatenated_active_rivalry_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Active rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active replay','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Active replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Passive rivalry','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_rivalry_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Passive rivalry' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Passive replay' for splitIndex in range(numSections)]+['inferredSwitches_Active rivalry','inferredSwitches_Active replay','inferredSwitches_Passive rivalry','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_joint_inferred_split_post'
# regressors=[['inferredSwitchesSPLITBYcovInfSwitchInfIntervalPost'+str(splitIndex)+'_Any' for splitIndex in range(numSections)]+['keyPress_Any','trialStarts_Any','saccades_Any','blinks_Any']]
# covariates=[[[] for splitIndex in range(numSections)]+[[],[],[],[]]]
# plottedRegressorIndices=[range(numSections)]
# timeShifts=[[0 for splitIndex in range(numSections)]+[0,0,0,0]]
# alignmentInfo=[['OKN' for splitIndex in range(numSections)]+['tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]




###
# The following can probably be safely left commented out: not very informative
#
# plotTitle='_Concatenated_any_active_reported_covariates_switchduration'
# regressors=[['reportedSwitches_Any active','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepSwitchdur_Any active','reportedSwitches_Any active_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_physical_covariates_switchduration'
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','physicalSwitches_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covPhysSwitchPhysSwitchdur_Any replay','physicalSwitches_Any replay_contrastRegressor'],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_covariates_switchduration'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepSwitchdur_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_covariates_switchduration'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covRepSwitchRepSwitchdur_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_physical_covariates_switchduration'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covPhysSwitchPhysSwitchdur_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_physical_covariates_switchduration'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],['covPhysSwitchPhysSwitchdur_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_active_reported_covariates_switchduration_withinPerceptZ'
# regressors=[['reportedSwitches_Any active','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepSwitchdurZscorePerPercept_Any active','reportedSwitches_Any active_contrastRegressor'],[],[],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['keys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_any_replay_physical_covariates_switchduration_withinPerceptZ'
# regressors=[['reportedSwitches_Active rivalry','inferredSwitches_Passive rivalry','physicalSwitches_Any replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],['covPhysSwitchPhysSwitchdurZscorePerPercept_Any replay','physicalSwitches_Any replay_contrastRegressor'],[],[],[],[],[]]]
# plottedRegressorIndices=[[2]]
# timeShifts=[[0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_rivalry_reported_covariates_switchduration_withinPerceptZ'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[['covRepSwitchRepSwitchdurZscorePerPercept_Active rivalry'],[],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[0]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['key','tevz','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_reported_covariates_switchduration_withinPerceptZ'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use reported switches and only look at active rivalry and active replay
# regressors=[['reportedSwitches_Active rivalry','reportedSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covRepSwitchRepSwitchdurZscorePerPercept_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','key','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_active_replay_physical_covariates_switchduration_withinPerceptZ'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],['covPhysSwitchPhysSwitchdurZscorePerPercept_Active replay'],[],
# [],[],[],[],[],[]]]
# plottedRegressorIndices=[[1]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','phys','tevz','tevz','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]
#
# plotTitle='_Concatenated_passive_replay_physical_covariates_switchduration_withinPerceptZ'    #pretend you don't know any better and use whatever time the switch is marked as (can be halfway a true transition), but use physical switches and only look at replay
# regressors=[['inferredSwitches_Active rivalry','physicalSwitches_Active replay','inferredSwitches_Passive rivalry','physicalSwitches_Passive replay','trialStarts_Any','probeReports_Any','unreportedProbes_Any','saccades_Any','blinks_Any']]
# covariates=[[[],[],[],['covPhysSwitchPhysSwitchdurZscorePerPercept_Passive replay'],[],[],[],[],[]]]
# plottedRegressorIndices=[[3]]
# timeShifts=[[0,0,0,0,0,0,0,0,0]]
# alignmentInfo=[['tevz','tevz','tevz','phys','tevz','tevz','tevz','tevz','tevz']]
# calculate_var_explained=False
# plotDictList=plotDictList+[{'plotTitle':plotTitle,'regressors':regressors,'plottedRegressorIndices':plottedRegressorIndices,'timeShifts':timeShifts,'calculateVarExplained':calculate_var_explained,'alignmentInfo':alignmentInfo,'covariates':covariates}]



#-----------------
#and then run the GLMS

plotColors=['r','g','b','c', 'm', 'y', 'k',(.25,.25,.25),(.5,.5,.5),(.8,.8,.8)]

for onePlotDict in plotDictList:
	
	if '_split_' in onePlotDict['plotTitle']:
		splitNumFilenameInsert='_'+str(numSections)+'sections_'
	else:
		splitNumFilenameInsert=''
		
	# if 'derivative' in onePlotDict['plotTitle']:
	# 	deriveIt=True
	# 	if 'zscore' in onePlotDict['plotTitle']:
	# 		zscoreIt=True
	# 		ylabel='z-score of deriv. of pupil area.'
	# 	else:
	# 		zscoreIt=False
	# 		ylabel='Derivative of pupil area (AU)'
	# else:
	# 	deriveIt=False
	# 	if 'zscore' in onePlotDict['plotTitle']:
	# 		zscoreIt=True
	# 		ylabel='z-score of pupil area.'
	# 	else:
	# 		zscoreIt=False
	# 		ylabel='Pupil area (AU)'
			
	#this wasn't changed until oct 14 2020 so wrong y-axis labels on earlier plots		
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

	for glmIndex in range(len(onePlotDict['regressors'])):
		individualsIncludedNew=[]
		for oneObserver in individualsIncluded:
			thisInfoDict=[oneInfoDict for oneInfoDict in concatenatedDictList if oneInfoDict['obs']==oneObserver][0]
			theseCovariateNames=onePlotDict['covariates'][glmIndex]
			theseCovariates=[[thisInfoDict[oneCovariateName] for oneCovariateName in covariateNamesOneRegressor] for covariateNamesOneRegressor in theseCovariateNames]
			numberOfNans=sum([sum([sum([math.isnan(item) for item in subsubList]) for subsubList in subList]) for subList in theseCovariates])	#check if any nans among the covariates. This happens if an observer only reports transitions of 0 duration in a given condition. 
			if numberOfNans>0:
				print 'Not including observer '+oneObserver+' for plot '+onePlotDict['plotTitle']+' because NaNs in covariates (probably because conditions with only 0-duration transitions).'
			else:
				individualsIncludedNew=individualsIncludedNew+[oneObserver]
				
		individualsIncluded=individualsIncludedNew[:]
	
	allDecoResponses=[]
	allStErrs=[]
	allPerObsData=[]
	
	allDecoResponsesCov=[]
	allStErrsCov=[]
	allPerObsDataCov=[]
	
	allCumDists=[]
	
	for glmIndex in range(len(onePlotDict['regressors'])):
		
		allDecoResponsesOneGLM=[]
		allCovariateResponsesOneGLM=[]
		allCumDistsOneGLM=[]
		
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
			
			theseCovariateNames=onePlotDict['covariates'][glmIndex]
			theseCovariates=[[thisInfoDict[oneCovariateName] for oneCovariateName in covariateNamesOneRegressor] for covariateNamesOneRegressor in theseCovariateNames]
			
			theseCovariatesNoZscore=[]
			for covariateNamesOneRegressor in theseCovariateNames:
				theseCovariatesNoZscoreOneRegressor=[]
				for oneCovariateName in covariateNamesOneRegressor:
					if 'contrastRegressor' in oneCovariateName:
						theseCovariatesNoZscoreOneRegressor=theseCovariatesNoZscoreOneRegressor+['n/a']
					else:
						if 'ZscorePerPercept_' in oneCovariateName:
							keyName=oneCovariateName.split('ZscorePerPercept_')[0]+'NoZscore_'+oneCovariateName.split('ZscorePerPercept_')[1]	
						else:
							keyName=oneCovariateName.split('_')[0]+'NoZscore_'+oneCovariateName.split('_')[1]	
						theseCovariatesNoZscoreOneRegressor=theseCovariatesNoZscoreOneRegressor+[thisInfoDict[keyName]]
				
				theseCovariatesNoZscore=theseCovariatesNoZscore+[theseCovariatesNoZscoreOneRegressor]
				
			eventTypesIndicesIncluded=[]
			for oneEventTypeIndex in range(len(theseEventNames)):
				if len(theseEvents[oneEventTypeIndex])>minNumEvs:
					eventTypesIndicesIncluded=eventTypesIndicesIncluded+[oneEventTypeIndex]
				else:
					print 'Event named '+theseEventNames[oneEventTypeIndex]+' excluded from '+onePlotDict['plotTitle']+' for observer '+oneObserver+' because insufficient events'
			
			theseEventsActuallyUsed=[theseEvents[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]			
			theseEventNamesActuallyUsed=[theseEventNames[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]	
			
			theseCovariatesActuallyUsed=[theseCovariates[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]			
			
			numberOfNans=sum([sum([sum([math.isnan(item) for item in subsubList]) for subsubList in subList]) for subList in theseCovariates])	#check if any nans among the covariates. This happens if an observer only reports transitions of 0 duration in a given condition. 
		
			theseCovariateNamesActuallyUsed=[theseCovariateNames[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]
		
			theseCovariatesNoZscoreActuallyUsed=[theseCovariatesNoZscore[eventTypeIndex] for eventTypeIndex in eventTypesIndicesIncluded]

			theseCovariatesActuallyUsedAsDictList=[]
			theseCovariatesNoZscoreActuallyUsedAsDictList=[]
			for regressorIndex in range(len(theseCovariatesActuallyUsed)):
				thisDict={}
				thisDictNoZscore={}
				for covariateIndex in range(len(theseCovariatesActuallyUsed[regressorIndex])):
				
					thisCovariateName=theseCovariateNamesActuallyUsed[regressorIndex][covariateIndex]
					thisDict[thisCovariateName]=theseCovariatesActuallyUsed[regressorIndex][covariateIndex]
				
					if 'ZscorePerPercept_' in thisCovariateName:
						thisCovariateNameNoZscore=thisCovariateName.split('ZscorePerPercept_')[0]+'NoZscore_'+thisCovariateName.split('ZscorePerPercept_')[1]	
					else:
						thisCovariateNameNoZscore=thisCovariateName.split('_')[0]+'NoZscore_'+thisCovariateName.split('_')[1]
					
					thisDictNoZscore[thisCovariateNameNoZscore]=theseCovariatesNoZscoreActuallyUsed[regressorIndex][covariateIndex]
				
				theseCovariatesActuallyUsedAsDictList=theseCovariatesActuallyUsedAsDictList+[thisDict]
				theseCovariatesNoZscoreActuallyUsedAsDictList=theseCovariatesNoZscoreActuallyUsedAsDictList+[thisDictNoZscore]
		
			#------------
		
			rfy = nideconv.ResponseFitter(input_signal=paddedPupilSamplesCleaned,sample_rate=newSampleRate)
		
			preCumDists={}
			postCumDists={}
		
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
				
				if theseCovariatesActuallyUsedAsDictList[eventIndex]=={}:
				
					rfy.add_event(event_name=eventName,onset_times=theseEventsActuallyUsed[eventIndex],basis_set='fourier',interval=thisInterval,n_regressors=numFuncs)
					preCumDists[eventName]=[[],[]]
					postCumDists[eventName]=[[],[]]
				
				else:
					covariatesActuallyUsedAsDictListPlusInterceptThisEvent=copy.deepcopy(theseCovariatesActuallyUsedAsDictList[eventIndex])
					covariatesActuallyUsedAsDictListPlusInterceptThisEvent['theIntercept']=numpy.ones(len(theseCovariatesActuallyUsedAsDictList[eventIndex][theseCovariateNamesActuallyUsed[eventIndex][0]]))
				
					rfy.add_event(event_name=eventName,onset_times=theseEventsActuallyUsed[eventIndex],basis_set='fourier',interval=thisInterval,n_regressors=numFuncs,covariates=covariatesActuallyUsedAsDictListPlusInterceptThisEvent)

					for covDictKey in theseCovariatesActuallyUsedAsDictList[eventIndex].keys():
					
						if 'ZscorePerPercept_' in covDictKey:
							keyNameNoZscore=covDictKey.split('ZscorePerPercept_')[0]+'NoZscore_'+covDictKey.split('ZscorePerPercept_')[1]	
							matchingSuffix='ZscorePerPercept_'
						else:
							keyNameNoZscore=covDictKey.split('_')[0]+'NoZscore_'+covDictKey.split('_')[1]
							matchingSuffix='_'
					
						if 'Pre'+matchingSuffix in covDictKey:
							CDFx=[-element for element in theseCovariatesNoZscoreActuallyUsedAsDictList[eventIndex][keyNameNoZscore]]
							CDFx.sort()
							CDFy=[1.-float(index)/len(CDFx) for index in range(len(CDFx))]
							preCumDists[eventName]=[CDFx,CDFy]
						elif 'Post'+matchingSuffix in covDictKey:
							CDFx=theseCovariatesNoZscoreActuallyUsedAsDictList[eventIndex][keyNameNoZscore][:]
							CDFx.sort()
							CDFy=[float(index)/len(CDFx) for index in range(len(CDFx))]
							postCumDists[eventName]=[CDFx,CDFy]				
				
			print 'About to regress GLM number '+str(glmIndex)+' of plot '+onePlotDict['plotTitle']+' for observer '+oneObserver+'.'
		
			rfy.regress()
			myIRFs=rfy.get_timecourses()
			#------------
		
			decoResponsesOneGLMAndObsInCorrectOrder=[]
			covariateResponsesOneGLMAndObsInCorrectOrder=[]
			cumDistsOneGLMandObsInCorrectOrder=[]
		
			for oneEventName in theseEventNames:
			
				thisEventIndex=[index for index in range(len(theseEventNames)) if theseEventNames[index]==oneEventName][0]
				covariateNamesThisEvent=theseCovariateNames[thisEventIndex]
				
				covariateResponsesThisEvent=[]
				cumDistsThisEvent=[]
			
				if not oneEventName in theseEventNamesActuallyUsed:
					decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]]
				
					for oneCovariateName in covariateNamesThisEvent:
						covariateResponsesThisEvent=covariateResponsesThisEvent+[[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]]
						cumDistsThisEvent=cumDistsThisEvent+[[],[]]
					
				else:
				
					if len(covariateNamesThisEvent)==0:
						decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[list(myIRFs[0][oneEventName])]
					else:
						decoResponsesOneGLMAndObsInCorrectOrder=decoResponsesOneGLMAndObsInCorrectOrder+[list(myIRFs[0][oneEventName]['theIntercept'])]
				
					for oneCovariateName in covariateNamesThisEvent:
						covariateResponsesThisEvent=covariateResponsesThisEvent+[list(myIRFs[0][oneEventName][oneCovariateName])]
				
					try:	
						cumDistsThisEvent=cumDistsThisEvent+[preCumDists[oneEventName]]	
					except KeyError:	#if this is a GLM without 'pre' covariates, then that key doesn't exist
						pass
					try:
						cumDistsThisEvent=cumDistsThisEvent+[postCumDists[oneEventName]]	
					except KeyError:
						pass
									
				covariateResponsesOneGLMAndObsInCorrectOrder=covariateResponsesOneGLMAndObsInCorrectOrder+[covariateResponsesThisEvent]
				cumDistsOneGLMandObsInCorrectOrder=cumDistsOneGLMandObsInCorrectOrder+[cumDistsThisEvent]

			allDecoResponsesOneGLM=allDecoResponsesOneGLM+[decoResponsesOneGLMAndObsInCorrectOrder]
			allCovariateResponsesOneGLM=allCovariateResponsesOneGLM+[covariateResponsesOneGLMAndObsInCorrectOrder]
			allCumDistsOneGLM=allCumDistsOneGLM+[cumDistsOneGLMandObsInCorrectOrder]
			
		theAverage=[]
		theStErr=[]
		thePerObsData=[]
		
		theAverageCov=[]
		theStErrCov=[]
		thePerObsDataCov=[]
		theAverageCumDist=[]
		
		for regressorIndex in range(len(allDecoResponsesOneGLM[0])):
			onlyIncludedObsIndices=[obsIndex for obsIndex in range(len(allDecoResponsesOneGLM)) if not (min(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1 and max(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1)]
			onlyIncludedObservers=[allDecoResponsesOneGLM[obsIndex][regressorIndex] for obsIndex in onlyIncludedObsIndices]
			
			if onlyIncludedObservers==[]:
				thisAverage=[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]
				thisAveragePreCum=[[]]
				thisAveragePostCum=[[]]
			else:
				thisAverage=numpy.average(onlyIncludedObservers,0)
				thisStErr=numpy.std(onlyIncludedObservers,0)/float(numpy.sqrt(len(onlyIncludedObservers)))
				
				if not(allCumDistsOneGLM[onlyIncludedObsIndices[0]][regressorIndex]==[]):	#this is what happens if this is not a pre- or post covariate but an 'other' one
					
					onlyIncludedObserversPreCum=[allCumDistsOneGLM[obsIndex][regressorIndex][0] for obsIndex in onlyIncludedObsIndices]
					onlyIncludedObserversPostCum=[allCumDistsOneGLM[obsIndex][regressorIndex][1] for obsIndex in onlyIncludedObsIndices]
					
					precumXflattened=[item for sublist in [element[0] for element in onlyIncludedObserversPreCum] for item in sublist]
					precumXflattened.sort()
					precumYflattened=[1.-float(index)/len(precumXflattened) for index in range(len(precumXflattened))]
					thisAveragePreCum=[precumXflattened,precumYflattened]
				
					postcumXflattened=[item for sublist in [element[0] for element in onlyIncludedObserversPostCum] for item in sublist]
					postcumXflattened.sort()
					postcumYflattened=[float(index)/len(postcumXflattened) for index in range(len(postcumXflattened))]
					thisAveragePostCum=[postcumXflattened,postcumYflattened]
					
				else:
					
					thisAveragePreCum=[[]]
					thisAveragePostCum=[[]]
					
			theAverage=theAverage+[thisAverage]
			theStErr=theStErr+[thisStErr]		#theStErr is all the across-obs st Errs (one per regressor) within this GLM
			thePerObsData=thePerObsData+[onlyIncludedObservers]	#thePerObsData is, for this GLM, all the individual-observer data for included observers only. Nesting is: all regressors, all included observers, all timepoints
			
			theAverageCumDist=theAverageCumDist+[[thisAveragePreCum,thisAveragePostCum]]
			
			theAverageCovsThisRegressor=[]
			theStErrCovsThisRegressor=[]
			thePerObsDataCovsThisRegressor=[]
			
			for covariateIndex in range(len(allCovariateResponsesOneGLM[0][regressorIndex])):
				onlyIncludedObservers=[allCovariateResponsesOneGLM[obsIndex][regressorIndex][covariateIndex] for obsIndex in range(len(allDecoResponsesOneGLM)) if not (min(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1 and max(allDecoResponsesOneGLM[obsIndex][regressorIndex])==-1)]
				
				if onlyIncludedObservers==[]:
					thisAverageOneCov=[-1 for element in range(int((decoInterval[1]-decoInterval[0])*newSampleRate))]
				else:
					thisAverageOneCov=numpy.average(onlyIncludedObservers,0)
					thisStErrOneCov=numpy.std(onlyIncludedObservers,0)/float(numpy.sqrt(len(onlyIncludedObservers)))
					
				theAverageCovsThisRegressor=theAverageCovsThisRegressor+[thisAverageOneCov]
				theStErrCovsThisRegressor=theStErrCovsThisRegressor+[thisStErrOneCov]
				thePerObsDataCovsThisRegressor=thePerObsDataCovsThisRegressor+[onlyIncludedObservers]
			
			theAverageCov=theAverageCov+[theAverageCovsThisRegressor]
			theStErrCov=theStErrCov+[theStErrCovsThisRegressor]
			thePerObsDataCov=thePerObsDataCov+[thePerObsDataCovsThisRegressor]
			
		allDecoResponsesOneGLM=allDecoResponsesOneGLM+[theAverage]	#add theAverage as if it's just another participant
		allDecoResponses=allDecoResponses+[allDecoResponsesOneGLM]
		
		allStErrs=allStErrs+[theStErr]		#allStErrs is bunch of theStErr's, one for each GLM
		
		allPerObsData=allPerObsData+[thePerObsData]		#allPerObsData is bunch of thePerObsData's, one for each GLM. So nesting is: GLM, regressor, observer (only included ones), timepoint. It's very similar to allDecoResponsesOneGLM but nested in a different order and with individual observers removed if their data didn't include a particular regressor.

		allCovariateResponsesOneGLM=allCovariateResponsesOneGLM+[theAverageCov]
		allDecoResponsesCov=allDecoResponsesCov+[allCovariateResponsesOneGLM]
		
		allStErrsCov=allStErrsCov+[theStErrCov]
		
		allPerObsDataCov=allPerObsDataCov+[thePerObsDataCov]
		
		allCumDistsOneGLM=allCumDistsOneGLM+[theAverageCumDist]
		allCumDists=allCumDists+[allCumDistsOneGLM]
		
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
				
				if regressorName == 'saccades_Any':
					x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
				elif regressorName == 'blinks_Any':
					x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
				else:
					x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
				
				pl.plot(x, y, color=plotColors[colorCounter], label=regressorName)
				colorCounter=colorCounter+1

		if not observerIndex==len(individualsIncluded):
			forOutputAccompanyingPlot=forOutputAccompanyingPlot+[forOutputAccompanyingPlotOneObs]
			
		pl.xlabel('Time from event (s)')
		pl.ylabel('Pupil size')
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
			
			if regressorName == 'saccades_Any':
				x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
			elif regressorName == 'blinks_Any':
				x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
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
	pl.ylabel('Pupil size (z score)')
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
		
			if regressorName == 'saccades_Any':
				x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
			elif regressorName == 'blinks_Any':
				x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
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

	pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+splitNumFilenameInsert+'_nideconv_regressors.pdf')
	numpy.save(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+splitNumFilenameInsert+'_nideconv_regressors_thedata',forOutputAccompanyingPlot)
	pl.close()
	
	#-----------------and now a plot of the covariates-------
	
	forOutputAccompanyingPlot=[]
	f = pl.figure(figsize = (35,35))
	for observerIndex in range(len(individualsIncluded)+1):

		forOutputAccompanyingPlotOneObs=[]
		
		s=f.add_subplot(5,6,observerIndex+1)
		colorCounter=0
		for glmIndex in range(len(onePlotDict['regressors'])):
			theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
			
			maxY=0
			for regressorIndex in theseRegressorIndices:
				
				for covariateIndex in range(len(onePlotDict['covariates'][glmIndex][regressorIndex])):
					
					if not 'contrastRegressor' in onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]:
						y=allDecoResponsesCov[glmIndex][observerIndex][regressorIndex][covariateIndex]
						forOutputAccompanyingPlotOneObs=forOutputAccompanyingPlotOneObs+[y]
						
						maxY=max([maxY,max(y)])
						
						regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]
						covariateName=onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]
			
						if regressorName == 'saccades_Any':
							x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
						elif regressorName == 'blinks_Any':
							x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
						else:
							x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
			
						pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
							
						colorCounter=colorCounter+1

				if not(allCumDists[glmIndex][observerIndex][regressorIndex] in [[],[[[]], [[]]]]):
				
					xPreCum=allCumDists[glmIndex][observerIndex][regressorIndex][0][0]
					yPreCum=allCumDists[glmIndex][observerIndex][regressorIndex][0][1]
					
					xPostCum=allCumDists[glmIndex][observerIndex][regressorIndex][1][0]
					yPostCum=allCumDists[glmIndex][observerIndex][regressorIndex][1][1]
			
					if len(xPreCum)>0:
				
						xPreCum=[xPreCum[index] for index in range(0,len(xPreCum),5)]		#undersample because there can be very many data points
						yPreCum=[yPreCum[index]*maxY for index in range(0,len(yPreCum),5)]
				
						xPostCum=[xPostCum[index] for index in range(0,len(xPostCum),5)]
						yPostCum=[yPostCum[index]*maxY for index in range(0,len(yPostCum),5)]
				
						pl.plot(xPreCum, yPreCum,color='k', dashes=[6, 2])
						pl.plot(xPostCum, yPostCum,color='k', dashes=[6, 2])
						pl.plot([min([decoIntervalSacc[0],decoIntervalBlink[0],decoInterval[0]]), max([decoIntervalSacc[1],decoIntervalBlink[1],decoInterval[1]])],[maxY,maxY],color='k')		#line indicating where 100% is

		if not observerIndex==len(individualsIncluded):
			forOutputAccompanyingPlot=forOutputAccompanyingPlot+[forOutputAccompanyingPlotOneObs]
			
		pl.xlabel('Time from event (s)')
		pl.ylabel('Pupil size')
		pl.xlim(min([decoIntervalSacc[0],decoIntervalBlink[0],decoInterval[0]]), max([decoIntervalSacc[1],decoIntervalBlink[1],decoInterval[1]]))
		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
		#sn.despine(offset=10)
		s.set_title(individualsIncludedPlusAverage[observerIndex])

	pl.legend(loc=2)

	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
	colorCounter=0
	for glmIndex in range(len(onePlotDict['regressors'])):
		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
		for regressorIndex in theseRegressorIndices:
			
			for covariateIndex in range(len(onePlotDict['covariates'][glmIndex][regressorIndex])):
				
				if not 'contrastRegressor' in onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]:
				
					y=allDecoResponsesCov[glmIndex][-1][regressorIndex][covariateIndex]		#-1 will be the across-obs average
					stErrs=allStErrsCov[glmIndex][regressorIndex][covariateIndex]

					regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]
					covariateName=onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]

					if regressorName == 'saccades_Any':
						x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
					elif regressorName == 'blinks_Any':
						x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
					else:
						x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
			
					pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
			
					downSampledY=[y[index] for index in range(0,len(y),40)]
					downSampledX=[x[index] for index in range(0,len(y),40)]
					downSampledErr=[stErrs[index] for index in range(0,len(y),40)]
			
					pl.errorbar(downSampledX, downSampledY, yerr=downSampledErr, color=plotColors[colorCounter], ls='none')
			
					tTestpValsVs0=[ttest_1samp([allPerObsDataCov[glmIndex][regressorIndex][covariateIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(0,len(y),40)]
			
					xForSignificantOnes=[]
					yForSignificantOnes=[]
					for candidateIndex in range(len(downSampledY)):
						if tTestpValsVs0[candidateIndex]<.01:
							xForSignificantOnes=xForSignificantOnes+[downSampledX[candidateIndex]]
							yForSignificantOnes=yForSignificantOnes+[downSampledY[candidateIndex]]

					pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)

					colorCounter=colorCounter+1

	pl.xlabel('Time from event (s)')
	pl.ylabel('Pupil size (z score per z score)')
	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
	#sn.despine(offset=10)
	s.set_title('Average plus error bars')
	
	s=f.add_subplot(5,6,observerIndex+3)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
	colorCounter=0
	for glmIndex in range(len(onePlotDict['regressors'])):
		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
		for regressorIndex in theseRegressorIndices:
			
			for covariateIndex in range(len(onePlotDict['covariates'][glmIndex][regressorIndex])):
				
				if not 'contrastRegressor' in onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]:
				
					y=allDecoResponsesCov[glmIndex][-1][regressorIndex][covariateIndex]		#-1 will be the across-obs average
					stErrs=allStErrsCov[glmIndex][regressorIndex][covariateIndex]

					regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]
					covariateName=onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]

					if regressorName == 'saccades_Any':
						x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
					elif regressorName == 'blinks_Any':
						x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
					else:
						x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
			
					pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
			
					pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
					pl.plot(x, [y[thisIndex]-stErrs[thisIndex] for thisIndex in range(len(y))], color=plotColors[colorCounter], linewidth=1)
					pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
					pl.plot(x, [y[thisIndex]+stErrs[thisIndex] for thisIndex in range(len(y))], color=plotColors[colorCounter], linewidth=1)
			
					colorCounter=colorCounter+1

	pl.xlabel('Time from event (s)')
	pl.ylabel(ylabel)
	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
	#sn.despine(offset=10)
	s.set_title('Average plus error bars')
	
	pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+splitNumFilenameInsert+'_nideconv_covariates.pdf')
	numpy.save(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+splitNumFilenameInsert+'_nideconv_covariates_thedata',forOutputAccompanyingPlot)
	pl.close()
	
	#-----------------and now a plot of the contrast regressors (if any)-------
	
	f = pl.figure(figsize = (35,35))
	for observerIndex in range(len(individualsIncluded)+1):

		s=f.add_subplot(5,6,observerIndex+1)
		colorCounter=0
		for glmIndex in range(len(onePlotDict['regressors'])):
			theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
			for regressorIndex in theseRegressorIndices:
				
				for covariateIndex in range(len(onePlotDict['covariates'][glmIndex][regressorIndex])):
					
					if 'contrastRegressor' in onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]:
						y=allDecoResponsesCov[glmIndex][observerIndex][regressorIndex][covariateIndex]
			
						regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]
						covariateName=onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]

						if regressorName == 'saccades_Any':
							x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
						elif regressorName == 'blinks_Any':
							x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
						else:
							x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
			
						pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
						colorCounter=colorCounter+1

		pl.xlabel('Time from event (s)')
		pl.ylabel('Pupil size')
		pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
		#sn.despine(offset=10)
		s.set_title(individualsIncludedPlusAverage[observerIndex])

	pl.legend(loc=2)

	s=f.add_subplot(5,6,observerIndex+2)	#observerIndex here is inherited from that for loop we just finished, so this will just be the next panel; wherever we were.
	colorCounter=0
	for glmIndex in range(len(onePlotDict['regressors'])):
		theseRegressorIndices=onePlotDict['plottedRegressorIndices'][glmIndex]
		for regressorIndex in theseRegressorIndices:
			
			for covariateIndex in range(len(onePlotDict['covariates'][glmIndex][regressorIndex])):
				
				if 'contrastRegressor' in onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]:
				
					y=allDecoResponsesCov[glmIndex][-1][regressorIndex][covariateIndex]		#-1 will be the across-obs average
					stErrs=allStErrsCov[glmIndex][regressorIndex][covariateIndex]

					regressorName=onePlotDict['regressors'][glmIndex][regressorIndex]
					covariateName=onePlotDict['covariates'][glmIndex][regressorIndex][covariateIndex]+', alignment: '+onePlotDict['alignmentInfo'][glmIndex][regressorIndex]

					if regressorName == 'saccades_Any':
						x = numpy.linspace(decoIntervalSacc[0],decoIntervalSacc[1], len(y))
					elif regressorName == 'blinks_Any':
						x = numpy.linspace(decoIntervalBlink[0],decoIntervalBlink[1], len(y))
					else:
						x = numpy.linspace(decoInterval[0],decoInterval[1], len(y))
			
					pl.plot(x, y, color=plotColors[colorCounter], label=covariateName)
			
					downSampledY=[y[index] for index in range(0,len(y),40)]
					downSampledX=[x[index] for index in range(0,len(y),40)]
					downSampledErr=[stErrs[index] for index in range(0,len(y),40)]
			
					pl.errorbar(downSampledX, downSampledY, yerr=downSampledErr, color=plotColors[colorCounter], ls='none')
			
					tTestpValsVs0=[ttest_1samp([allPerObsDataCov[glmIndex][regressorIndex][covariateIndex][obsIndex][timePointIndex] for obsIndex in range(len(allPerObsData[glmIndex][regressorIndex]))],0)[1] for timePointIndex in range(0,len(y),40)]
			
					xForSignificantOnes=[]
					yForSignificantOnes=[]
					for candidateIndex in range(len(downSampledY)):
						if tTestpValsVs0[candidateIndex]<.01:
							xForSignificantOnes=xForSignificantOnes+[downSampledX[candidateIndex]]
							yForSignificantOnes=yForSignificantOnes+[downSampledY[candidateIndex]]

					pl.scatter(xForSignificantOnes, yForSignificantOnes,color='k',s=20)

					colorCounter=colorCounter+1

	pl.xlabel('Time from event (s)')
	pl.ylabel('Pupil size (z score, kind of)')
	pl.axhline(0,color = 'k', lw = 0.5, alpha = 0.5)
	#sn.despine(offset=10)
	s.set_title('Average plus error bars')
	if colorCounter>0:
		pl.savefig(myPath+figuresSubFolder+'/'+onePlotDict['plotTitle']+splitNumFilenameInsert+'_nideconv_contrastRegressors.pdf')	
		pl.close()
		
