import argparse
import glob
import os
import sys

import numpy as np

import DTWsystem

parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--feature", help="Selected features to use. (mfcc, posteriors, bottleneck, string, lattice)")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--system", required=True)
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

if arguments.feature.lower() in features:
    if arguments.verbose:
        print("The {} features were chosen.".format(arguments.feature))
    feature = arguments.feature
else:
    if arguments.verbose:
        print("MFCC features were chosen implicitly.")
    feature = 'mfcc'


def FeatureToExtension(feature):
    if feature == 'mfcc':
        return '*.wav'  # mozno by nebolo odveci to presunut tiez do nejakeho priecinku na rovnakej urovni
    elif feature == 'posteriors':
        return '*/*.lin'
    elif feature == 'bottleneck':
        return '*/*.bnf'  # este neviem ako to bude asi md5
    elif feature == 'string':
        return '*/*.txt'
    elif feature == 'lattice':
        return '*/*.latt'


def GetFiles(file, feature):
    return glob.glob(file + FeatureToExtension(feature))


def LabelFiles(src, feature, exact_path=None, label=None):
    if exact_path is None and label is None:
        test_target = np.array(GetFiles(src + 'eval/eval_goal/', feature))
        test_labels = np.array([1] * len(test_target))  # 1 obtain pre-recorded messsage
        test_clear = np.array(GetFiles(src + 'eval/eval_clear/', feature))
        test_labels = np.concatenate((test_labels, np.array([0] * len(test_clear))), axis=0)
        test_files = np.concatenate((test_target, test_clear), axis=0)

        train_target = np.array(GetFiles(src + 'train/train_goal/', feature))
        train_labels = np.array([1] * len(train_target))  # 1 obtain pre-recorded messsage
        train_clear = np.array(GetFiles(src + 'train/train_clear/', feature))
        train_labels = np.concatenate((train_labels, np.array([0] * len(train_clear))), axis=0)
        train_files = np.concatenate((train_target, train_clear), axis=0)

        return train_files, train_labels, test_files, test_labels
    elif exact_path is not None and label is not None:
        files = np.array(GetFiles(src + exact_path, feature))
        labels = np.array([label] * len(files))
        return files, labels
    else:
        raise Exception("Type both exact_path and label or leave it as None for LabelFiles funtion")


if __name__ == "__main__":
    system = arguments.system.lower()
    train_files, train_labels, test_files, test_labels = LabelFiles(src, feature)
    if system == 'basedtw':
        result_list = DTWsystem.BaseDtwUnknown([train_files, train_labels], [test_files, test_labels],
                                                        feature=feature, reduce_dimension=True)
        print(result_list)
