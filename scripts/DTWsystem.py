import librosa
import numpy as np
from numpy.core._multiarray_umath import ndarray
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import ArrayFromFeatures
from fastdtw import fastdtw
from sklearn.utils import shuffle
import time

from third_party_scripts.seg_dtw.sdtw import segmental_dtw as SegmentalDTW


def MyDTW(o, r):
    cost_matrix: ndarray = cdist(o, r, metric='euclidean')
    m, n = np.shape(cost_matrix)
    for i in range(m):
        for j in range(n):
            if (i == 0) & (j == 0):
                cost_matrix[i, j] = cost_matrix[i, j]  # inf
            elif i == 0:
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i, j - 1]  # inf
            elif j == 0:
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i - 1, j]  # inf
            else:
                cost_matrix[i, j] = np.min([cost_matrix[i, j] + cost_matrix[i - 1, j],
                                            cost_matrix[i, j] + cost_matrix[i, j - 1],
                                            cost_matrix[i, j] * np.sqrt(2) + cost_matrix[i - 1, j - 1]])
    # backtracking
    path = [m - 1], [n - 1]
    i, j = m - 1, n - 1
    while i != 0 or j != 0:
        backtrack = np.argmin([cost_matrix[i - 1, j - 1], cost_matrix[i - 1, j], cost_matrix[i, j - 1]])
        if backtrack == 1:
            path[0].append(i - 1)
            path[1].append(j)
            i -= 1
        elif backtrack == 2:
            path[0].append(i)
            path[1].append(j - 1)
            j -= 1
        else:
            path[0].append(i - 1)
            path[1].append(j - 1)
            i -= 1
            j -= 1
    np_path = np.array(path, dtype=np.int64)
    return cost_matrix, cost_matrix[-1, -1] / (cost_matrix.shape[0] + cost_matrix.shape[1]), np_path.T


def Similarity(wp):
    sim_list = []
    tmp_list = []
    false_trend = 0
    constant_false = 0
    for point in zip(*[iter(np.flip(wp, 0))] * 2):  # take two in a row
        if len(tmp_list) > 0 and (tmp_list[-1][0] == point[0][0] or tmp_list[-1][1] == point[0][1]):
            # false_trend += 1
            # constant_false += 1
            pass
        elif len(tmp_list) > 0:
            constant_false = 0
        if point[0][0] < point[1][0] and point[0][1] < point[1][1]:
            tmp_list.append(point[0])
            tmp_list.append(point[1])
            constant_false = 0
        else:
            tmp_list.append(point[0])
            tmp_list.append(point[1])
            false_trend += 1
            constant_false += 1
        # print(len(sim_list), len(tmp_list), false_trend, constant_false)

        if constant_false > 6:
            for i in range(constant_false):
                if len(tmp_list) > 0:
                    del tmp_list[-1]
            if (false_trend != 0 and
                len(tmp_list) / false_trend > 5) or \
                    false_trend == 0:
                sim_list.append(np.array(tmp_list))
            tmp_list = []
            false_trend = 0
            constant_false = 0

    if len(tmp_list) >= 100:
        sim_list.append(np.array(tmp_list))
    for i in range(len(sim_list) - 1, -1, -1):
        if len(sim_list[i]) < 100:
            sim_list.pop(i)
    sim_list = np.array(sim_list, dtype=object)
    return sim_list


def SimilarityNew(wp, interval=None):
    if interval is None:
        interval = [0.98, 1.02]
    sim_list = []
    tmp_list = []
    score_ratio = 0
    ratio_list = []
    good_trend = 0
    false_trend = 0
    constant_false = 0
    have_hit = False
    RIGHT = 0
    UP = 1
    DIAG = 2
    direction = -1  # unknown at first
    prev_direction = -1
    prev_point = None
    for point in wp:  # np.flip(wp, 0):
        if prev_point is None:
            prev_point = point
            continue
        # going RIGHT ->
        if prev_point[0] == point[0] and prev_point[1] < point[1]:
            direction = RIGHT
        # going UP ^
        elif prev_point[0] < point[0] and prev_point[1] == point[1]:
            direction = UP
        # going DIAG ⟋`
        else:
            direction = DIAG
        # print("PREVIOUS DIRECTION:", prev_direction, "    DIRECTION:", direction)
        if tmp_list:
            if (direction == RIGHT and prev_direction == UP) or (direction == UP and prev_direction == RIGHT):
                constant_false = 0
            # false_trend += 1
            elif (direction == RIGHT and prev_direction == RIGHT) or (direction == UP and prev_direction == UP):
                # good_trend -= 1
                false_trend += 1
                constant_false += 1
            elif direction == DIAG:
                good_trend += 1
                constant_false = 0
            else:
                constant_false = 0
            tmp_list.append(point)

        if tmp_list == [] and direction == DIAG:
            if prev_direction == -1 or (prev_direction == DIAG and good_trend >= 5):
                tmp_list.append(point)
            good_trend += 1
            false_trend = 0
            constant_false = 0
        if constant_false >= 25:
            del tmp_list[-1]
            false_trend -= 1
            for i in range(constant_false):
                if len(tmp_list) > 0:
                    del tmp_list[-1]
                    false_trend -= 1
            if len(tmp_list) >= 200:
                ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
                if score_ratio < (1 - np.absolute(ratio - 1)) and have_hit == False:
                    score_ratio = 1 - np.absolute(ratio - 1)  # finding maximum
                if interval[0] < ratio < interval[1]:
                    if not have_hit:
                        score_ratio = 1 - np.absolute(ratio - 1)
                        have_hit = True
                    if score_ratio > ratio:
                        score_ratio = 1 - np.absolute(ratio - 1)
                    sim_list.append(np.array([tmp_list[0], tmp_list[-1]]))
                    ratio_list.append("{:.4f}".format(ratio))
            constant_false = 0
            false_trend = 0
            good_trend = 0
            tmp_list.clear()
        prev_point = point
        prev_direction = direction

    if len(tmp_list) >= 200:
        del tmp_list[-1]
        false_trend -= 1
        for i in range(constant_false):
            if len(tmp_list) > 0:
                del tmp_list[-1]
                false_trend -= 1
        ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
        if interval[0] < ratio < interval[1]:
            sim_list.append(np.array([tmp_list[0], tmp_list[-1]]))
            ratio_list.append("{:.4f}".format(ratio))
    sim_list = np.array(sim_list, dtype=object)
    return tuple(sim_list), tuple(ratio_list), score_ratio, have_hit


def InitCheckShuffle(train, test):
    if test is None:
        test = []
        test_nested = []
    else:
        test[0], test[1] = shuffle(test[0], test[1], random_state=10)
        test_nested = test.copy()
    if train is None:
        train = []
        train_nested = []
    else:
        train[0], train[1] = shuffle(train[0], train[1], random_state=10)
        train_nested = train.copy()
    return train, train_nested, test, test_nested


def BaseDtwUnknown(train=None, test=None, feature='mfcc', reduce_dimension=True):
    train, train_nested, test, test_nested = InitCheckShuffle(train, test)
    result_list = []
    hits_count = 0
    threshold = [0.9, 1.1]
    hit_threshold = 0.0
    loop_count = 0
    if feature not in ['mfcc', 'posteriors', 'bottleneck']:
        raise Exception("For BaseDTW system use only mfcc, posteriors or bottleneck feature vectors")
    if feature == 'mfcc':
        hit_threshold = 35.0
        loop_count = 49
    if feature == 'posteriors':
        hit_threshold = 4.0
        loop_count = 49
    if feature == 'bottleneck':
        hit_threshold = 1.25
        loop_count = 99

    one_round = []
    for idx, file in enumerate(test[0]):  # train[0] == list of files
        start = time.time()
        parsed_file = ArrayFromFeatures.Parse(file)
        file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
        score_list = []
        dist_list = []
        hit_dist = []
        test_nested[0], test_nested[1] = shuffle(test_nested[0], test_nested[1], random_state=8)
        for idx_nested, file_nested in enumerate(test_nested[0]):
            if file.split('/')[-1] == file_nested.split('/')[-1]:  # same file
                continue
            parsed_file_nested = ArrayFromFeatures.Parse(file_nested)
            file_nested_array = ArrayFromFeatures.GetArray(file_nested, feature, reduce_dimension)
            final_dist, wp = fastdtw(file_array, file_nested_array, dist=euclidean)
            sim_list, ratio_list, score, hit = SimilarityNew(np.asarray(wp), threshold)
            final_dist = final_dist / (file_array.shape[0] + file_nested_array.shape[0])
            if sim_list:
                distances = []
                for point in sim_list:
                    dist, wp = fastdtw(file_array[point[0][0]:point[1][0]], file_nested_array[point[0][1]:point[1][1]],
                                       dist=euclidean)
                    distances.append(dist / (file_array[point[0][0]:point[1][0]].shape[0] +
                                             file_nested_array[point[0][1]:point[1][1]].shape[0]))
                final_dist = min(distances)
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                parsed_file[0], "0" if test[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(test[0]),
                parsed_file_nested[0], "0" if test_nested[1][idx_nested] == 0 else "1:" + parsed_file_nested[3],
                                                                                     idx_nested + 1, len(test[0]),
                final_dist))

            score_list.append(score)
            dist_list.append(final_dist)

            if final_dist < hit_threshold:
                hits_count += 1
                hit_dist.append(final_dist)
            # "surely" got hit, going to the next sample / or counting too much
            if hits_count >= 1 or idx_nested > loop_count:
                break
        f = open("evalBaseDTW_{}.txt".format(feature), "a")

        # if len(result_list_nested) > 0:
        if hits_count > 0:
            hits_count = 0
            score_list = list(filter(lambda x: threshold[0] < x < threshold[1], score_list))
            append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1",
                                                             min(score_list), max(hit_dist))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
        else:
            # if not result_list_nested:
            if not score_list:
                append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", "0.0", min(dist_list))
                result_list.append(append_string)
                f.write(append_string)
            else:
                append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0",
                                                                 max(score_list), min(dist_list))
                result_list.append(append_string)  # train[1] == label
                f.write(append_string)
        f.close()
        one_round.append(time.time() - start)
        print('Next File. Time: {}s'.format(one_round[-1]))
    ft = open("timeBaseDTW_{}.txt".format(feature), "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(one_round), np.sum(one_round)))
    ft.close()
    return result_list


def SecondPassDtw(data, rqa_list, hit_threshold, frame_reduction, feature, reduce_dimension, loop_count, rqa_time,
                  sdtw):
    dtw_time = []
    result_list = []
    for idx, file in enumerate(data[0]):
        start = time.time()
        parsed_file = ArrayFromFeatures.Parse(file)
        file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
        file_array = ArrayFromFeatures.ReduceFrames(file_array, size=frame_reduction)
        # file_array = ArrayFromFeatures.CompressFrames(file_array, size=frame_reduction)

        if float(parsed_file[-1]) < 3.0:  # skipping samples of total duration shorter than 3seconds
            print("Skipping", file.split('/')[-1], data[1][idx],
                  "because the total duration {} is less than 3.0 seconds".format(parsed_file[-1]))
            continue
        dist_list = []
        have_hit = False
        nested_loop_count = 0
        rqa_list = shuffle(rqa_list, random_state=8)
        # loop through candidates-> queries to search in 'file' 
        for idx_nested, rqa_item in enumerate(rqa_list):
            if rqa_item[1] == [] or len(rqa_item[1]) * rqa_item[2] < 450:  # skipping short frames
                continue
            if file.split('/')[-1] == rqa_item[0].split('/')[-1]:
                continue
            nested_loop_count += 1
            parsed_file_nested = ArrayFromFeatures.Parse(rqa_item[0])
            file_nested_array = ArrayFromFeatures.GetArray(rqa_item[0], feature, reduce_dimension)
            file_nested_array = ArrayFromFeatures.ReduceFrames(file_nested_array, size=frame_reduction)
            # file_nested_array = ArrayFromFeatures.CompressFrames(file_nested_array, size=frame_reduction)
            if sdtw:
                path = SegmentalDTW(file_nested_array[int(rqa_item[1][0][0]) * rqa_item[2] // frame_reduction:int(
                    rqa_item[1][-1][0]) * rqa_item[2] // frame_reduction], file_array, R=5, L=300 / frame_reduction,
                                    dist='cosine')
                # wp = np.asarray(path[1][3])*frame_reduction   # not necessary, uncomment when want to know the warping path
                dtw_distance = path[0]
            else:
                cost_matrix, wp = librosa.sequence.dtw(X=file_array.T,
                                                       Y=file_nested_array[int(rqa_item[1][0][0]) * rqa_item[
                                                           2] // frame_reduction:int(rqa_item[1][-1][0]) * rqa_item[
                                                           2] // frame_reduction].T,
                                                       metric='euclidean',
                                                       weights_mul=np.array([np.sqrt([2]), 1, 1],
                                                                            dtype=np.float64))  # cosine rychlejsie
                dtw_distance = cost_matrix[wp[-1, 0], wp[-1, 1]]
            if dtw_distance is None:
                continue
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(data[0]),
                parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3],
                                                                                     idx_nested + 1, len(rqa_list),
                dtw_distance))
            if dtw_distance < hit_threshold:
                have_hit = True
                f = open("evalRQA_{}_unknown_{}.txt".format("SDTW" if sdtw == True else "DTW", feature), "a")
                append_string = "{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "1", dtw_distance)
                result_list.append(append_string)  # train[1] == label
                f.write(append_string)
                f.close()
                break
            else:
                dist_list.append(dtw_distance)
            if nested_loop_count > loop_count:  # after loop_count is exceeded going to next file
                break
        if have_hit == False:
            f = open("evalRQA_{}_unknown_{}.txt".format("SDTW" if sdtw == True else "DTW", feature), "a")
            append_string = "{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "0", min(dist_list))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
            f.close()
        dtw_time.append(time.time() - start)
        print('Next File. Time: {}s'.format(dtw_time[-1]))
    print(np.sum(dtw_time), np.mean(dtw_time))
    ft = open("timeRQA_{}_unknown_{}.txt".format("SDTW" if sdtw == True else "DTW", feature), "a")
    ft.write("RQA_MEAN:{}\tRQA_TOTAL:{}\tDTW_MEAN:{}\tDTW_TOTAL:{}\tTOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(
        np.mean(rqa_time), np.sum(rqa_time),
        np.mean(dtw_time), np.sum(dtw_time),
        np.mean(rqa_time) + np.mean(dtw_time), np.sum(rqa_time) + np.sum(dtw_time)))
    ft.close()
    return result_list
