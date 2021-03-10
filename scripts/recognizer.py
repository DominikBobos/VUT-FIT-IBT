import os, glob, sys 
import argparse, pathlib
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--feature", help="Selected features to use. (mfcc, posteriors, bottleneck, string, lattice)")
parser.add_argument("-v", "--verbose", action="store_true")
arguments = parser.parse_args()

features = ['mfcc', 'posteriors', 'bottleneck', 'string', 'lattice']

if not arguments.src:
	if arguments.verbose:
		print("Current directory '{}' is chosen as source directory".format(os.getcwd))
	src = os.getcwd()
else:
	if arguments.verbose:
		print("Directory '{}' is chosen as source directory".format(arguments.src))
	src = arguments.src

if arguments.feature in features:
	if arguments.verbose:
		print("The {} features were chosen.".format(arguments.feature))
	feature = arguments.feature
else:
	if arguments.verbose:
		print("MFCC features were chosen implicitly.")
	feature = 'mfcc'


def FeatureToExtension(feature):
	if feature == 'mfcc':
		return '*.wav'
	elif feature == 'posteriors':
		return '*/*.lin'
	elif feature == 'bottleneck':
		return '*/*.bnf' #este neviem ako to bude asi md5
	elif feature == 'string':
		return '*/*.txt'
	elif feature == 'lattice':
		return '*/*.latt'

def GetFiles(file, feature):
	return glob.glob(file + FeatureToExtension(feature))

def LabelFiles(src, feature):

	test_target = np.array(GetFiles(src + 'eval/eval_goal/', feature))
	test_clear = np.array([1] * len(test_target))  # 1 obtain pre-recorded messsage
	t = np.array(GetFiles(src + 'eval/eval_clear/', feature))
	test_clear = np.concatenate((test_clear, np.array([0] * len(t))), axis=0)
	test_target = np.concatenate((test_target, t), axis=0)

	train_target = np.array(GetFiles(src + 'train/train_goal/', feature))
	train_clear = np.array([1] * len(train_target))  # 1 obtain pre-recorded messsage
	t = np.array(GetFiles(src + 'train/train_clear/', feature))
	train_clear = np.concatenate((train_clear, np.array([0] * len(t))), axis=0)
	train_target = np.concatenate((train_target, t), axis=0)

	return train_target, train_clear, test_target, test_clear

train_target, train_clear, test_target, test_clear = LabelFiles(src, feature)
print(train_target, train_clear, test_target, test_clear)