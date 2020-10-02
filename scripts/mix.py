##
#   @file mix.py
#   @author Dominik Bobos 
#   @brief Script for mixing pre-recorded messages with speech examples
#

from pydub import AudioSegment          # for splitting, mixing audio files
from pydub import effects               # for changing speed
from librosa.core import load           # for loading audio files
import math                             # for rounding numbers
import argparse, textwrap               # for parsing arguments
import os, sys, glob, re                # for file path, finding files, etc
import random as rand 	                # for random numbers generator
import numpy
from collections import Counter         # for statistics


##TODO -make comments

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Arguments parsing
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
parser = argparse.ArgumentParser( formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-m", "--mode", help=textwrap.dedent("""Specifies which type of pre-recorded message you want to use.
  Possible values are: A, B, C"""))
parser.add_argument("-g", "--gain", action="store_true", help="Change volume gain of the pre-recorded message")                            
parser.add_argument("-s", "--speed", action="store_true", help="Change speed of the pre-recorded message")                       
parser.add_argument("-r", "--repeat", action="store_true", help="Cut or repeat the pre-recorded messages")    
parser.add_argument("--onlymodify", action="store_true", help="Just modify the operator message and export it")     
parser.add_argument("--mpath", help="Path to the directory with the pre-recorded messages, --mpath=PATH")  
parser.add_argument("--spath", help="Path to the directory with the speech .wav files, --spath=PATH")   
parser.add_argument("--export", help="Path to the directory where to export the mixed audio files")
parser.add_argument("--stats", action="store_true", help="show statistics about the wav files from the given directory from arg mpath=PATH")
parser.add_argument("--lt", help="Will use longer recordings than specified in seconds")    # use audio longer than specified in seconds
parser.add_argument("--st", help="Will use shorter recordings than specified in seconds")   # use audio shorter than specified in seconds
parser.add_argument("--random", default=0, help="""Select severity of randomness
Possible values are:
  0 = no random values, use the original values of the pre-recorded message
  1 = low differences, e.g. gain between -3dB+3dB, speed 0.95-1.05, repeat 1-10.0
  2 = optimal differences, e.g. gain between -6dB+6dB, speed 0.9-1.1, repeat 0.8-30.0
  3 = high differences, e.g. gain between -10dB+6dB, speed 0.85-1.15, repeat 0.7-40.0""") 
arguments = parser.parse_args()

if arguments.mode and not re.match(r"^(a|b|c|A|B|C)([0-9][0-9])$|^(a|b|c|A|B|C)$", arguments.mode): 
	sys.stderr.write("Invalid mode! Use help for more information\n")
	sys.exit(1)
elif arguments.mode != None:
	mode = arguments.mode.upper() 
	print(mode)
else:
	mode = ''

if arguments.random:
	try:
		if int(arguments.random) != 0 and \
		int(arguments.random) != 1 and \
		int(arguments.random) != 2 and \
		int(arguments.random) != 3:
			sys.stderr.write("Invalid random value! Use help for more information\n")
			sys.exit(1)
		else:
			random = int(arguments.random)
	except:
		sys.stderr.write("Invalid random value! Use help for more information\n")
		sys.exit(1)
else:
	random = 0 

if not arguments.mpath:
	msgsrc = os.getcwd()
else:
	if os.path.isdir(arguments.mpath):
		msgsrc = arguments.mpath
	else:
		sys.stderr.write("Invalid mpath directory.\n")
		sys.exit(1)

if not arguments.spath:
	recsrc = os.getcwd()
else:
	if os.path.isdir(arguments.spath):
		recsrc = arguments.spath
	else:
		sys.stderr.write("Invalid spath directory.\n")
		sys.exit(1)

if not arguments.export:
	expsrc = os.getcwd()
else:
	if os.path.isdir(arguments.export):
		expsrc = arguments.export
	else:
		try:
			os.mkdir(arguments.export)                                  # create destination directory
			expsrc = arguments.export
		except:
			sys.stderr.write("Invalid directory for export.\n")
			sys.exit(1)

if random == 0:
	GAIN_RAND = [0.0, 0.0]
	SPEED_RAND = [1.0, 1.0]
	REPEAT_RAND = [1.0, 1.0]
elif random == 1:
	GAIN_RAND = [-3.0, +3.0]
	SPEED_RAND = [0.95 , 1.05]
	REPEAT_RAND = [1.0, 10.0]
elif random == 2:
	GAIN_RAND = [-6.0, +6.0]
	SPEED_RAND = [0.9, 1.10]
	REPEAT_RAND = [0.8, 30.0]
elif random == 3:
	GAIN_RAND = [-10.0, +6.0]
	SPEED_RAND = [0.85, 1.15]
	REPEAT_RAND = [0.7, 40.0]



"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Gets duration in seconds from the given audio file
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
def get_duration(file):
	return AudioSegment.from_wav(file).duration_seconds


def modify(file):
	mixedFile = AudioSegment.from_wav(file)
	gRand, rRand, sRand = 0,1,1
	if arguments.speed and random != 0:
		sRand = round(rand.uniform(SPEED_RAND[0], SPEED_RAND[1]),2)
		mixedFile = mixedFile._spawn(mixedFile.raw_data, overrides={"frame_rate": int(mixedFile.frame_rate * sRand)})
	if arguments.gain and random != 0:
		gRand = round(rand.uniform(GAIN_RAND[0], GAIN_RAND[1]),2)
		mixedFile += gRand
	if arguments.repeat and random != 0:
		rRand = round(rand.uniform(REPEAT_RAND[0], REPEAT_RAND[1]),2)
		floorInt = math.floor(rRand)
		left = rRand - floorInt
		if floorInt != 0: 
			mixedFile = mixedFile * floorInt + mixedFile[:math.ceil(left*get_duration(file)*1000)]
		else:
			mixedFile = mixedFile[:math.ceil(left*get_duration(file)*1000)]
	
	return file, mixedFile, mixedFile.duration_seconds ,gRand, rRand, sRand
		# mixedFile.export(arguments.export + file.split('/')[-1], format="wav")


def get_type(file):
	try:
		fileType = file.split('/')
		fileType = fileType[-1][0] #first letter shows type of the pre-recorded message
	except:
		fileType = file[0]
	return fileType


def make_filename(recordFile, messageFile, position):
	filename = recordFile.split('/')[-1].split('.w')[0] + '__' + messageFile[0].split('/')[-1].split('.w')[0] + \
		'_ST(' + "{:.2f}".format(position) + ')L(' + "{:.2f}".format(messageFile[2]) + ')G(' + \
		"{:.2f}".format(messageFile[3]) + ')R(' + "{:.2f}".format(messageFile[4]) + \
		')S(' + "{:.2f}".format(messageFile[5]) + ').wav'
	return filename


def mix():
	msg = glob.glob(msgsrc + mode + '*.wav')
	rec = glob.glob(recsrc + '*.wav')	
	index = 0
	current = 0
	msgCount = len(msg)
	recCount = len(rec)
	if arguments.onlymodify:
		for file in msg:
			current += 1	
			modified = modify(file)
			filename = make_filename("path/sw00000-A_0_0.wav", modified, 0)
			print("Exporting {}/{}: {}:".format(current, msgCount, filename))
			modified[1].export(arguments.export + filename, format="wav")
	else:
		for file in rec:
			current += 1
			# print("Modifying:", msg[index])
			modified = modify(msg[index])
			#filter out unwanted files
			if arguments.lt and int(arguments.lt) > get_duration(file):
				continue
			if arguments.st and int(arguments.st) < get_duration(file):
				continue
			record = AudioSegment.from_wav(file)
			if get_type(modified[0]) == 'A':		#add to the beginning
				record = modified[1] + record 
				position = 0
			elif get_type(modified[0]) == 'B':		#add somewhere in between
				randCut = math.ceil(rand.uniform(0.0, get_duration(file)))
				record = record[:randCut*1000] + modified[1] + record[randCut*1000:]
				position = randCut
			else:	#type 'C'	            #add at the end
				record = record + modified[1] 
				position = get_duration(file)

			# name it the right way and export
			filename = make_filename(file, modified, position)

			print("Exporting {}/{}: {}:".format(current, recCount, filename))
			record.export(arguments.export + filename, format="wav")
			
			index += 1
			if index == msgCount:
				index = 0


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Counts the total count of all files
counts the total duration of all files
orders and counts the occurrences of files
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
def count(files):
	statList = []
	total, count, msgTotal, gainTotal, repeatTotal, speedTotal = 0, 0, 0, 0, 0, 0
	for file in files:
		if arguments.lt and int(arguments.lt) > get_duration(file):
			continue
		elif arguments.st and int(arguments.st) < get_duration(file):
			continue
		else:
			try:
				msg = file.split('__')[1]
				msgType, msgInfo = msg.split('_')
				start, length, gain, repeat, speed, extension = msgInfo.split(')')
			except:
				pass
			current = get_duration(file)
			count += 1
			total += current
			msgTotal += float(length.split('(')[1])
			gainTotal += float(gain.split('(')[1])
			repeatTotal += float(repeat.split('(')[1])
			speedTotal += float(speed.split('(')[1])
			statList.append(msgType)
	stat = Counter(statList)
	return total, stat, count, msgTotal, gainTotal, repeatTotal, speedTotal



"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Shows statistics about audio files in a given directory
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
def stats():
	msg = glob.glob(msgsrc + mode + '*.wav')
	if len(msg) != 0:
		ret = count(msg)
		srcTotal, srcStat, srcCount, msgStats = ret[0], ret[1], ret[2], ret[3:]
		avgLength = msgStats[0] / srcCount
		avgGain = msgStats[1] / srcCount
		avgRepeat = msgStats[2] / srcCount
		avgSpeed = msgStats[3] / srcCount
		print("____________________________________________________________________________________")
		print("\nTotal count of all recordings:  \t", srcCount)
		print("Total length of records: \t", "{:.2f}".format(srcTotal), "[s] |", 
			"{:.2f}".format(srcTotal/60), "[m] |", "{:.2f}".format(srcTotal/3600), "[h]")
		print("Total length of the mixed pre-recorded messages: \t", "{:.2f}".format(msgStats[0]), "[s] |", 
			"{:.2f}".format(msgStats[0]/60), "[m] |", "{:.2f}".format(msgStats[0]/3600), "[h]")
		print("The Average length of the mixed pre-recorded messages: \t", "{:.2f}".format(avgLength), "[s]")
		print("The Average gain: \t", "{:.2f}".format(avgGain), "[dB]")
		print("The Average loop: \t", "{:.2f}".format(avgRepeat), "times") 
		print("The Average speed: \t", "{:.2f}".format(avgSpeed), "to original \n") 
		for item in srcStat.most_common():
			print("Message type: \t", item[0], "  \t Count: ", item[1])
		print("____________________________________________________________________________________\n")
	else:
		print("No statistics available.")


if not arguments.stats:
	mix()
else:
	stats()
