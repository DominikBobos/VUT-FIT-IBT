import librosa
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import ArrayFromFeatures
from fastdtw import fastdtw
from sklearn.utils import shuffle
import time

def MyDTW(o, r):
    cost_matrix = cdist(o, r, metric='euclidean')
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
                min_cost = np.min([cost_matrix[i, j] + cost_matrix[i - 1, j],
                                   cost_matrix[i, j] + cost_matrix[i, j - 1],
                                   cost_matrix[i, j] * np.sqrt(2) + cost_matrix[i - 1, j - 1]])
                cost_matrix[i, j] = min_cost
    # backtracking
    path = [m - 1], [n - 1]
    i, j = m - 1, n - 1
    while (True):
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
        if i == 0 and j == 0:
            break
    np_path = np.array(path, dtype=np.int)
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
    for point in wp:#np.flip(wp, 0):
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
                    score_ratio = 1 - np.absolute(ratio - 1)# finding maximum
                if interval[0] < ratio < interval[1]:
                    if have_hit == False:
                        score_ratio = 1 - np.absolute(ratio -1)
                        have_hit = True
                    if score_ratio > ratio:
                        score_ratio = 1 - np.absolute(ratio -1)
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


def GramMatrix(feature):
    matrix = feature.dot(
        feature.T)  # (feature.dot(feature.T)) / (np.linalg.norm(feature) * np.linalg.norm(feature.T))
    return 0.5 * (matrix + 1)


# return feature


def ImageFilter(matrix, threshold=0.7, percentile=70, variance=5):
    diag_matrix = np.zeros((100, 100), np.int)
    # np.fill_diagonal(np.fliplr(a), 1)  # flip
    np.fill_diagonal(diag_matrix, 1)

    matrix[matrix < threshold] = 0.0
    matrix[matrix >= threshold] = 1.0
    matrix = ndimage.percentile_filter(matrix, percentile=percentile, footprint=diag_matrix, mode='constant', cval=0.0)
    matrix = ndimage.gaussian_filter(matrix, variance)
    return matrix


def BaseDtwUnknown(train=None, test=None, feature='mfcc', reduce_dimension=True):
    if test is None:
        test = []
    else:
        test[0], test[1] = shuffle(test[0], test[1], random_state=10)
        test_nested = test.copy()
    if train is None:
        train = []
    else:
        train[0], train[1] = shuffle(train[0], train[1], random_state=10)
    result_list = []
    hits_count = 0
    threshold = [0.9, 1.1]
    hit_threshold = 0.0
    loop_count = 0
    if feature == 'mfcc':
        hit_threshold = 35.0
        loop_count = 49
    if feature == 'posteriors':
        hit_threshold = 4.0
        loop_count = 49
    if feature == 'bottleneck':
        hit_threshold = 1.25 #netusim zatial
        loop_count = 99

    
    one_round = []
    for idx, file in enumerate(test[0]):  # train[0] == list of files
        # if idx > 124:   # first 124 are done
        #     continue

        # if idx < 125:   # starting from 125 to 249
        #     continue
        # if idx > 249:
        #     continue
        
        # if idx < 250:   # from 250 to the end
        #     continue

        start = time.time()
        parsed_file = ArrayFromFeatures.Parse(file)
        file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
        # result_list_nested = []
        score = 0 
        score_list = []
        dist_list = []
        hit_dist = []
        test_nested[0], test_nested[1] = shuffle(test_nested[0], test_nested[1], random_state=8)
        for idx_nested, file_nested in enumerate(test_nested[0]):
            if file.split('/')[-1] == file_nested.split('/')[-1]:  # same file
                continue
            parsed_file_nested = ArrayFromFeatures.Parse(file_nested)
            file_nested_array = ArrayFromFeatures.GetArray(file_nested, feature, reduce_dimension)
            # final_dist, wp = librosa.sequence.dtw(X=file_array.T, Y=file_nested_array.T, metric='euclidean',
            #                                        weights_mul=np.array([np.sqrt([2]), 1, 1],
            #                                                             dtype=np.float))  # cosine rychlejsie
            final_dist, wp = fastdtw(file_array, file_nested_array, dist=euclidean)
            sim_list, ratio_list, score, hit = SimilarityNew(np.asarray(wp), threshold)
            final_dist = final_dist / (file_array.shape[0] + file_nested_array.shape[0])
            if sim_list:
                distances = []
                for point in sim_list:
                    dist, wp = fastdtw(file_array[point[0][0]:point[1][0]], file_nested_array[point[0][1]:point[1][1]], dist=euclidean)
                    distances.append(dist/(file_array[point[0][0]:point[1][0]].shape[0]+file_nested_array[point[0][1]:point[1][1]].shape[0]))
                final_dist = min(distances)
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                parsed_file[0], "0" if test[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(test[0]),
                parsed_file_nested[0], "0" if test_nested[1][idx_nested] == 0 else "1:" + parsed_file_nested[3], idx_nested + 1, len(test[0]),
                final_dist)) # final_dist[wp[-1, 0], wp[-1, 1]]
            
            # del final_dist
            # del wp
            # del parsed_file_nested
            # del file_nested_array

            score_list.append(score)
            dist_list.append(final_dist)

            if final_dist < hit_threshold:
                hits_count += 1
                hit_dist.append(final_dist)

            # if len(ratio_list) > 0:
            #     # score_list.append(score)
            #     # result_list_nested.append([file_nested.split('/')[-1], sim_list, ratio_list, score])
            #     if hit:
            #         hits_count += 1
            if hits_count >= 1 or idx_nested > loop_count:  # "surely" got hit, going to the next sample / or counting too much
                break
        f = open("evalBaseDTW.txt", "a")

        # if len(result_list_nested) > 0:
        if hits_count > 0:
            hits_count = 0
            score_list = list(filter(lambda x: threshold[0]< x < threshold[1], score_list))

            # result_list.append([file.split('/')[-1], test[1][idx], "1", min(x[2] for x in result_list_nested)[0],
            #                     tuple(result_list_nested)])  # train[1] == label
            f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1",
                                              min(score_list), max(hit_dist)))
                                              # min(x[2] for x in result_list_nested)[0]))
        else:
            # if not result_list_nested:
            if not score_list:
                # result_list.append(
                #     [file.split('/')[-1], test[1][idx], "0", 0.0, tuple(result_list_nested)])  # train[1] == label
                f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", "0.0", min(dist_list)))
            else:
                # result_list.append([file.split('/')[-1], test[1][idx], "0", max(x[2] for x in result_list_nested)[0],
                #                     tuple(result_list_nested)])  # train[1] == label
                f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0",
                                                  max(score_list), min(dist_list)))
                                                  # max(x[2] for x in result_list_nested)[0]))
        # result_list_nested.clear()
        del score_list
        del parsed_file
        del file_array
        del hit_dist
        del dist_list
        # print(result_list)
        f.close()
        one_round.append(time.time()-start)
        print('Next File. Time: {}s'.format(one_round[-1]))
    ft = open("timeBaseDTW.txt", "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(one_round),np.sum(one_round)))
    ft.close()
    return result_list
