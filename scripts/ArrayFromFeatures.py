from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from HTK import HTKFile
import numpy as np


def GetMFCC(file, verbose=False):
	sample_f, signal = wavfile.read(file)
	mfcc = mfcc(signal, sample_f, 0.025, 0.01, 13, 512)
	if verbose == True:
		print("FILE | SAMPLE | NUMPY ARRAY SHAPE")
		print(file, sample_f, mfcc.shape, sep=" | ")
	return mfcc


def LoadHTK(file, verbose=False):
	htk_reader = HTKFile()
	htk_reader.load(file)
	feature = np.array(htk_reader.data)
	
	if verbose == True:
		print("FILE | SAMPLES NUM | FEATURES NUM | SAMPLE PERIOD | QUALIFIERS")
		print(file, htk_reader.nSamples, htk_reader.nFeatures, htk_reader.sampPeriod, htk_reader.qualifiers, sep=" | ")
		print("FILE | NUMPY ARRAY SHAPE")
		print(file, feature.shape, sep=" | ")
	return feature

def ReduceDimension(feature):
	reduced = np.empty([feature.shape[0], feature.shape[1] // 3])

	for frame in range(feature.shape[0]):
		for i in range(0, feature.shape[1], 3):
			reduced[frame, i//3] = (feature[frame, i] + feature[frame, i+1] + feature[frame, i+2])
	print(reduced.shape)
	return reduced


def Parse(filename):
	parsed_file = []
	del_path = ''
	temp = ''
	try:
		del_path = filename.split('/')[-1]
	except IndexError:
		del_path = filename
	temp = del_path.split('_')
	parsed_file.append(temp[0])	 # sample ID name
	parsed_file.append(int(temp[1]))  # part of cut sample
	# <20 -> sample without message 
	if len(del_path) < 20:
		parsed_file.append(float(temp[2][:-4]))			# sample duration in seconds
		parsed_file.append(temp[2][-3:])		# extension format (wav, lin etc.)
		parsed_file.append(float(temp[2][:-4]))	# total duration
		print(parsed_file)
		return parsed_file
	else:
		parsed_file.append(float(temp[2]))  # sample duration in seconds
	temp = del_path.split('__')[-1]		# part with operator message info
	parsed_file.append(temp.split('_')[0])	# operator message ID
	temp = temp.split('_')[-1].split('(')
	parsed_file.append(float(temp[1].split(')')[0]))	# message start point
	parsed_file.append(float(temp[2].split(')')[0]))	# message duration in seconds
	parsed_file.append(float(temp[3].split(')')[0]))	# volume gain
	parsed_file.append(float(temp[4].split(')')[0]))	# repeat count
	parsed_file.append(float(temp[5].split(')')[0]))	# speed
	parsed_file.append(temp[5].split('.')[-1])			# extension format (wav, lin etc.)
	parsed_file.append(parsed_file[2] + parsed_file[5])	# total duration of mixed file
	# [sampleID, cutPart, sampleDuration, messageID, messStart, messDuration, volume, repeat, speed, extension, totalDuration]
	return parsed_file	  	
