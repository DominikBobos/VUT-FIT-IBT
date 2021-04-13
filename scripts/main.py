import argparse
import glob
import os
import sys

import numpy as np

import DTWsystem
import RQAsystem
import FuzzyStringSystem


def CheckPositive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is an invalid positive int value for argument --frame-reduction".format(value))
    return ivalue

parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--feature", help="Features to use. (mfcc, posteriors, bottleneck, string)")
parser.add_argument("--cluster-feature", help="Features to use. (mfcc, posteriors, bottleneck)")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--system", required=True, help="Available systems: basedtw, arenjansen/rqa_unknown, rqa_dtw_unknown/2pass_dtw_unknown, rqa_sdtw_unknown/2pass_sdtw_unknown,")
parser.add_argument("--frame-reduction", type=CheckPositive, help="Downsampling the the feature vector, averaging given N-frames")
arguments = parser.parse_args()

features = ['mfcc', 'posteriors', 'bottleneck', 'string']
cluster_features = ['mfcc', 'posteriors', 'bottleneck']

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
    print("MFCC features were chosen implicitly.")
    feature = 'mfcc'

if arguments.cluster_feature in features:
    if arguments.verbose:
        print("The {} cluster features were chosen.".format(arguments.cluster_feature))
    cluster_feature = arguments.cluster_feature
else:    
    print("MFCC features were chosen implicitly.")
    cluster_feature = 'mfcc'




def FeatureToExtension(feature):
    if feature == 'mfcc':
        return '*.wav'  # mozno by nebolo odveci to presunut tiez do nejakeho priecinku na rovnakej urovni
    elif feature == 'posteriors':
        return '*/*.lin'
    elif feature == 'bottleneck':
        return '*/*.fea'  # este neviem ako to bude asi md5
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
        files0 = np.array(GetFiles(exact_path[0], feature))
        labels = np.array([label[0]] * len(files0))
        files1 = np.array(GetFiles(exact_path[1], feature))
        labels = np.concatenate((labels, np.array([0] * len(files1))), axis=0)
        files = np.concatenate((files0, files1), axis=0)
        return files, labels
    else:
        raise Exception("Type both exact_paths and labels (target first, second non-target) or leave it as None for LabelFiles funtion")


if __name__ == "__main__":
    system = arguments.system.lower()
    train_files, train_labels, test_files, test_labels = LabelFiles(src, feature)
    frame_reduction = arguments.frame_reduction if arguments.frame_reduction is not None else 1
    if system == 'basedtw':
        result_list = DTWsystem.BaseDtwUnknown([train_files, train_labels], [test_files, test_labels],
                                                        feature=feature, reduce_dimension=True)
    if system == 'arenjansen' or system == 'rqa_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True)
    if system == 'rqa_dtw_unknown' or system == '2pass_dtw_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True)
    if system == 'rqa_sdtw_unknown' or system == '2pass_sdtw_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True, sdtw=True)
    if system == 'rqacluster_dtw_unknown' or system == '2pass_cluster_dtw_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True, cluster=True, metric='euclidean')
    if system == 'rqacluster_sdtw_unknown' or system == '2pass_cluster_sdtw_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True, sdtw=True, cluster=True, metric='cosine')
    if system == 'cluster_dtw_known':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True, sdtw=False, cluster=True, metric='cosine', known=True)
    if system == 'cluster_sdtw_known':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                        feature=feature, 
                                        frame_reduction=frame_reduction, reduce_dimension=True, 
                                        second_pass=True, sdtw=True, cluster=True, metric='cosine', known=True)
    if system == 'fuzzy_match_base':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels], system=system)
    if system == 'fuzzy_match_pau_analysis':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels], system=system)
    if system == 'rqacluster_fuzzy_match_unknown':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels], system=system, clust_feature=cluster_feature)
    if system == 'rqacluster_fuzzy_match_known':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels], system=system, clust_feature=cluster_feature)