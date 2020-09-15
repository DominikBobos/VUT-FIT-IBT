##
#   @file script.py
#   @author Dominik Bobos 
#   @brief Script for splitting audio files into multiple parts and showing statistics
#

from pydub import AudioSegment
from librosa.core import load
import math
import argparse
import os, sys, glob
from collections import Counter         # for statistics

#ffmpeg needed just mark it somewhere


parser = argparse.ArgumentParser()
parser.add_argument("--sec", help="duration in minutes of the wanted cut length")     # how long cuttings we want in minutes
parser.add_argument("--src", help="source directory path")                            # directory to import
parser.add_argument("--dst", help="destination directory path")                       # directory to export
parser.add_argument("--stats", action="store_true", 
    help="show statistics about the wav files in either source or destination directory")
parser.add_argument("--lt", help="Will cut longer recordings than specified in seconds")
parser.add_argument("--st", help="Will cut shorter recordings than specified in seconds")
parser.add_argument("--rm", action="store_true", help="Will remove the original file after splitting it")
arguments = parser.parse_args()

if not arguments.sec and not arguments.stats:
    sys.stderr.write("Please specify the duration in seconds of the wanted cut length using --sec=NUMBER .\n")
    sys.exit(1)
if arguments.sec:
    splitSec = int(arguments.sec)
if not arguments.src:
    src = os.getcwd()
else:
    if os.path.isdir(arguments.src):
        src = arguments.src
    else:
        sys.stderr.write("Invalid source directory.\n")
        sys.exit(1)
if not arguments.dst:
    dst = os.getcwd()
else:
    if os.path.isdir(arguments.dst):
        dst = arguments.dst
    else:
        try:
            os.mkdir(arguments.dst)                                  # create destination directory
            dst = arguments.dst
        except:
            sys.stderr.write("Invalid destination directory.\n")
            sys.exit(1)


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Loads audio files from source directory 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
def get_audio_file(file):
    return [load(f) for f in glob.glob(file)]

def get_duration(file):
    return AudioSegment.from_wav(file).duration_seconds
    
def single_split(fromSec, toSec, splitFilename, file):
    t1 = fromSec * 1000
    t2 = toSec * 1000
    splitAudio = AudioSegment.from_wav(file + '.wav')[t1:t2]
    splitFilename = splitFilename.split("/")[-1]
    splitAudio.export(dst + splitFilename, format="wav")
    
def multiple_split(splitSec):
    files = glob.glob(src+'*.wav')
    for file in files:  
        order = 0
        if arguments.lt and int(arguments.lt) > get_duration(file):
            continue
        if arguments.st and int(arguments.st) < get_duration(file):
            continue
        try: 
            file.split('_')
            order = int(file.split('_')[1])
        except:
            pass
        totalSec = get_duration(file)
        try:
            file.split('.')[2]
            file = file.split('.')[0] + '.' + file.split('.')[1]
        except:
            file = file.split('.')[0]
        for i in range(0, math.ceil(totalSec), splitSec):
            if i >= math.ceil(totalSec)-splitSec:
                seconds = get_duration(file + '.wav')
                seconds = seconds - i
                splitFilename = file.split('_')[0] + '_' + str(order + i//splitSec) + '_' + "{:.2f}".format(seconds) + ".wav" 
            else:
                splitFilename = file.split('_')[0] + '_' + str(order + i//splitSec) + '_' + str(splitSec) + ".wav"
            single_split(i, i+splitSec, splitFilename, file)
            print(splitFilename.split('/')[-1], ' : ' ,str(i//splitSec) + ' Done')
            if i >= math.ceil(totalSec)-splitSec:
                print('Splited successfully\n')
        if arguments.rm and get_duration(file + '.wav') > splitSec:
            print('Removing file: ', file + ".wav")
            os.remove(file + '.wav')

def count(files):
    statList = []
    total = 0
    count = 0
    for file in files:
        if arguments.lt and int(arguments.lt) > get_duration(file):
            continue
        elif arguments.st and int(arguments.st) < get_duration(file):
            continue
        else:
            current = get_duration(file)
            count += 1
            total += current
            statList.append(current)
    stat = Counter(statList)
    return total, stat, count

def stats():
    srcFiles = glob.glob(src+'*.wav')
    dstFiles = glob.glob(dst+'*.wav')
    if len(srcFiles) != 0:
        srcTotal, srcStat, srcCount = count(srcFiles)
        print("____________________________________________________________________________________")
        print("\nTotal count of all source recordings:  \t", srcCount)
        print("Total length of source records: \t", "{:.2f}".format(srcTotal), "[s] |", 
            "{:.2f}".format(srcTotal/60), "[m] |", "{:.2f}".format(srcTotal/3600), "[h]", "\n")
        for item in srcStat.most_common():
            print("Records lengths [sec]: \t", item[0], "  \t Count: ", item[1])
        print("____________________________________________________________________________________\n")
    if len(dstFiles) != 0:
        dstTotal, dstStat, dstCount = count(dstFiles)
        print("____________________________________________________________________________________")
        print("\nTotal count of destination recordings:  \t", dstCount)
        print("Total length of destination records: \t", "{:.2f}".format(dstTotal), "[sec] |", 
            "{:.2f}".format(dstTotal/60), "[min] |", "{:.2f}".format(dstTotal/3600), "[h]", "\n")
        for item in dstStat.most_common():
            print("Records lengths [sec]: \t", item[0], "  \t Count: ", item[1])
        print("____________________________________________________________________________________\n")
    if len(srcFiles) == 0 and len(dstFiles) == 0:
        print("No statistics available.")

if not arguments.stats:
    multiple_split(splitSec)
if arguments.stats:
    stats()


# from pydub import AudioSegment
# from librosa.core import load
# import math
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--min")
# arguments = parser.parse_args()
# splitMin = arguments.system

# class SplitWavAudioMubin():
#     def __init__(self, folder, filename):
#         self.folder = folder
#         self.filename = filename
#         self.filepath = folder + '/' + filename
        
#         self.audio = AudioSegment.from_wav(self.filepath)

#     def get_audio_file(file):
#         return [load(f) for f in glob.glob(file)]
    
#     def get_duration(self):
#         return self.audio.duration_seconds
    
#     def single_split(self, from_min, to_min, split_filename):
#         t1 = from_min * 60 * 1000
#         t2 = to_min * 60 * 1000
#         split_audio = self.audio[t1:t2]
#         split_audio.export(self.folder + '/' + split_filename, format="wav")
        
#     def multiple_split(self, splitMin):
#         total_mins = math.ceil(self.get_duration() / 60)
#         for i in range(0, total_mins, splitMin):
#             if i == total_mins - splitMin:
#                 seconds = math.ceil(self.get_duration())
#                 seconds = seconds - (total_mins-1)*60
#                 split_fn = self.filename + '_' + str(i) + '_' + "{:.2f}".format(seconds / 60)
#             else:
#                 split_fn = self.filename + '_' + str(i) + '_' + str(splitMin)
#             self.single_split(i, i+splitMin, split_fn)
#             print(str(i) + ' Done')
#             if i == total_mins - splitMin:
#                 print('All splited successfully')

# folder = '../'
# file = 'sw02005-A.wav'
# split_wav = SplitWavAudioMubin(folder, file)
# split_wav.multiple_split(splitMin)