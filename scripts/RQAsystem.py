import librosa
import numpy as np
import time

import ArrayFromFeatures
import DTWsystem


def RqaAnalysis(data, rqa_threshold, frame_reduction, feature, reduce_dimension, second_pass):
    rqa_list = []
    rqa_time = []
    result_list = []
    for idx, file in enumerate(data[0]):  # rqa analysis first
        start = time.time()
        parsed_file = ArrayFromFeatures.Parse(file)
        if float(parsed_file[-1]) < 3.0:  # shorter samples are skipped 
            rqa_list.append([file, [], frame_reduction, 10000.0])  # giving high score to that one
            print("Skipping", file.split('/')[-1], data[1][idx],
                  "because the total duration {} is less than 3.0 seconds".format(parsed_file[-1]),
                  time.time() - start)
            rqa_time.append(time.time() - start)
            if not second_pass:
                f = open("evalRQA_{}.txt".format(feature), "a")
                append_string = "{}\t{}\t{}\t{}\t{}\n".format(
                    file.split('/')[-1],
                    data[1][idx],
                    "0",
                    "0",
                    rqa_list[-1][2])
                result_list.append(append_string)
                f.write(append_string)
                f.close()
            continue
        file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
        reduced_array = np.compress(
            [True if i % frame_reduction == 0 else False for i in range(file_array.shape[0])], file_array, axis=0)
        rec_matrix = librosa.segment.recurrence_matrix(reduced_array.T, width=30, k=reduced_array.shape[0] // 10,
                                                       mode='affinity',
                                                       metric='cosine')  # ['connectivity', 'distance', 'affinity']
        score, path = librosa.sequence.rqa(rec_matrix, np.inf, np.inf, knight_moves=True)
        path_length = (int(path[-1][0]) - int(path[0][0])) * frame_reduction
        # only frames longer than 2.5 seconds are valid and giving high score penalty for them
        rqa_list.append([file, path, frame_reduction, np.sum(rec_matrix[path]) / (int(path[-1][0]) - int(path[0][0]))]
                        if path_length > 250 else [file, [], frame_reduction, 5000.0])
        rqa_time.append(time.time() - start)
        print("Processing {}[{}] ({}/{}), path length: {}, score: {}, time: {}".format(
            parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3],
            idx + 1, len(data[0]),
            path_length,
            rqa_list[-1][2],
            time.time() - start))
        if not second_pass:
            f = open("evalRQA_unknown_{}.txt".format(feature), "a")
            append_string = "{}\t{}\t{}\t{}\t{}\n".format(
                file.split('/')[-1],
                data[1][idx],
                "0" if (rqa_list[-1][1] == []) or (rqa_list[-1][2] > rqa_threshold) else "1",
                path_length,
                rqa_list[-1][2])
            result_list.append(append_string)
            f.write(append_string)
            f.close()
    print("RQA mean:", np.mean(rqa_time), "RQA total:", np.sum(rqa_time))
    if not second_pass:
        ft = open("timeRQA_unknown_{}.txt".format(feature), "a")
        ft.write("TOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(np.mean(rqa_time), np.sum(rqa_time)))
        ft.close()
    return rqa_list, rqa_time, result_list


def RqaDtwUnknown(train=None, test=None, feature='mfcc', frame_reduction=1, reduce_dimension=True, second_pass=False,
                  sdtw=False):
    if frame_reduction < 1:
        raise ValueError("Frame reduction must be at least 1 or bigger")
    train, train_nested, test, test_nested = DTWsystem.InitCheckShuffle(train, test)
    result_list = []
    rqa_time = []
    hit_threshold = 0.0
    loop_count = 50
    rqa_threshold = 500.0
    if feature not in ['mfcc', 'posteriors', 'bottleneck']:
        raise Exception("For BaseDTW system use only mfcc, posteriors or bottleneck feature vectors")
    if feature == 'mfcc':
        hit_threshold = 0.004
    if feature == 'posteriors':
        hit_threshold = 9.0
        if sdtw:
            hit_threshold = 0.3
    if feature == 'bottleneck':
        hit_threshold = 1.25
        if sdtw:
            hit_threshold = 0.28

    rqa_list = ArrayFromFeatures.OpenPickle("evaluations/objects/rqa_list_{}.pkl".format(feature))

    if not rqa_list:
        rqa_list, rqa_time, result_list = RqaAnalysis(data=test, rqa_threshold=rqa_threshold,
                                                      frame_reduction=frame_reduction,
                                                      feature=feature, reduce_dimension=reduce_dimension,
                                                      second_pass=second_pass)
    if not rqa_time:
        rqa_time = [0.]

    ArrayFromFeatures.SavePickle("evaluations/objects/rqa_list_{}.pkl".format(feature), rqa_list)

    if second_pass:
        result_list = DTWsystem.SecondPassDtw(data=test, rqa_list=rqa_list, hit_threshold=hit_threshold,
                                              frame_reduction=frame_reduction, feature=feature,
                                              reduce_dimension=reduce_dimension,
                                              loop_count=loop_count, rqa_time=rqa_time, sdtw=sdtw)

    return result_list
