import librosa
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import ArrayFromFeatures
from fastdtw import fastdtw
from sklearn.utils import shuffle
import time
import pickle

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
        train_nested = train.copy()
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
        hit_threshold = 1.25 
        loop_count = 99

    
    one_round = []
    for idx, file in enumerate(test[0]):  # train[0] == list of files
        # to separate to  3 threads
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

            score_list.append(score)
            dist_list.append(final_dist)

            if final_dist < hit_threshold:
                hits_count += 1
                hit_dist.append(final_dist)

            if hits_count >= 1 or idx_nested > loop_count:  # "surely" got hit, going to the next sample / or counting too much
                break
        f = open("evalBaseDTW_{}.txt".format(feature), "a")

        # if len(result_list_nested) > 0:
        if hits_count > 0:
            hits_count = 0
            score_list = list(filter(lambda x: threshold[0]< x < threshold[1], score_list))

            result_list.append("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1",
                                              min(score_list), max(hit_dist)))  # train[1] == label
            f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1",
                                              min(score_list), max(hit_dist)))
        else:
            # if not result_list_nested:
            if not score_list:
                result_list.append("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", "0.0", min(dist_list)))
                f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", "0.0", min(dist_list)))
            else:
                result_list.append("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0",
                                                  max(score_list), min(dist_list)))# train[1] == label
                f.write("{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0",
                                                  max(score_list), min(dist_list)))                               
        f.close()
        one_round.append(time.time()-start)
        print('Next File. Time: {}s'.format(one_round[-1]))
    ft = open("timeBaseDTW_{}.txt".format(feature), "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(one_round),np.sum(one_round)))
    ft.close()
    return result_list


def RQAunknown(train=None, test=None, feature='mfcc', frame_reduction=5, reduce_dimension=True, second_pass=True):
    if frame_reduction < 1:
        raise ValueError("Frame reduction must be at least 1 or bigger")
    if test is None:
        test = []
    else:
        test[0], test[1] = shuffle(test[0], test[1], random_state=10)
        test_nested = test.copy()
    if train is None:
        train = []
    else:
        train[0], train[1] = shuffle(train[0], train[1], random_state=10)
        train_nested = train.copy()
    result_list = []
    rqa_list = []
    rqa_time = []
    dtw_time = []
    hit_threshold = 0.0
    loop_count = 0
    rqa_threshold = 500.0
    if feature == 'mfcc':
        hit_threshold = 35.0
        loop_count = 49
    if feature == 'posteriors':
        hit_threshold = 9.0 #no idea
        loop_count = 49
    if feature == 'bottleneck':
        hit_threshold = 1.25 #netusim zatial
        loop_count = 99

    try:
        open_file = open("evaluations/objects/rqa_list_{}.pkl".format(feature), "rb")
        rqa_list = pickle.load(open_file)
        open_file.close()
        print("rqa_list_{}.pkl found. RQA analysis will not be performed".format(feature))
    except IOError:
        print("No rqa_list_{}.pkl exist. RQA analysis will be performed".format(feature))

    if not rqa_list:
        for idx, file in enumerate(test[0]):    #rqa analysis first
            start = time.time()
            parsed_file = ArrayFromFeatures.Parse(file)
            if float(parsed_file[-1]) < 3.0:  # total duration
                rqa_list.append([file, [], 10000.0])
                print("Skipping", file.split('/')[-1], test[1][idx], "because the total duration {} is less than 3.0 seconds".format(parsed_file[-1]), time.time()-start)
                rqa_time.append(time.time()-start)
                if second_pass == False:
                    f = open("evalRQA_{}.txt".format(feature), "a")
                    result_list.append("{}\t{}\t{}\t{}\t{}\n".format(
                                        file.split('/')[-1], 
                                        test[1][idx], 
                                        "0",
                                        0, 
                                        rqa_list[-1][2])) 
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(
                                        file.split('/')[-1], 
                                        test[1][idx], 
                                        "0", 
                                        "0",
                                        rqa_list[-1][2])) 
                    f.close()
                continue
            file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
            reduced_array = np.compress([True if i % frame_reduction == 0 else False for i in range(file_array.shape[0])], file_array, axis=0)
            rec_matrix = librosa.segment.recurrence_matrix(reduced_array.T,width=30, k=reduced_array.shape[0]//10, mode='affinity', metric='cosine') #['connectivity', 'distance', 'affinity']
            score, path = librosa.sequence.rqa(rec_matrix, np.inf, np.inf, knight_moves=True)
            path_length = (int(path[-1][0])-int(path[0][0]))*frame_reduction
            rqa_list.append([file, path, np.sum(rec_matrix[path])/(int(path[-1][0])-int(path[0][0]))] if path_length > 250 else [file, [], 5000.0]) #only frames longer than 2.5 seconds are valid
            rqa_time.append(time.time()-start)
            print("Processing {}[{}] ({}/{}), path length: {}, score: {}, time: {}".format(
                    parsed_file[0], "0" if test[1][idx] == 0 else "1:" + parsed_file[3], 
                    idx + 1, len(test[0]), 
                    path_length,
                    rqa_list[-1][2], 
                    time.time()-start))
            if second_pass == False:
                f = open("evalRQA_{}.txt".format(feature), "a")
                result_list.append("{}\t{}\t{}\t{}\t{}\n".format(
                                    file.split('/')[-1], 
                                    test[1][idx], 
                                    "0" if (rqa_list[-1][1] == []) or (rqa_list[-1][2] > rqa_threshold) else "1", 
                                    path_length,
                                    rqa_list[-1][2])) 
                f.write("{}\t{}\t{}\t{}\t{}\n".format(
                                    file.split('/')[-1], 
                                    test[1][idx], 
                                    "0" if (rqa_list[-1][1] == []) or (rqa_list[-1][2] > rqa_threshold) else "1", 
                                    path_length,
                                    rqa_list[-1][2])) 
                f.close()
    if not rqa_time:
        rqa_time = [0.]
    print("RQA mean:", np.mean(rqa_time), "RQA total:", np.sum(rqa_time))
    if second_pass == False:
        ft = open("timeRQA_{}.txt".format(feature), "a")
        ft.write("TOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(np.mean(rqa_time), np.sum(rqa_time)))
        ft.close()

    open_file = open("evaluations/objects/rqa_list_{}.pkl".format(feature), "wb")
    pickle.dump(rqa_list, open_file)
    open_file.close()
    

    if second_pass:
        result_list = []
        for idx, file in enumerate(test[0]):
            start = time.time()
            parsed_file = ArrayFromFeatures.Parse(file)
            file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
            file_array = np.compress([True if i % frame_reduction == 0 else False for i in range(file_array.shape[0])], file_array, axis=0)
            # result_list_nested = []
            dist_list = []
            have_hit = False
            nested_loop_count = 0
            rqa_list = shuffle(rqa_list, random_state=8)
            for idx_nested, rqa_item in enumerate(rqa_list):
                if rqa_item[1] == []:
                    continue
                if file.split('/')[-1] == rqa_item[0].split('/')[-1]:
                    continue
                nested_loop_count += 1
                parsed_file_nested = ArrayFromFeatures.Parse(rqa_item[0])
                file_nested_array = ArrayFromFeatures.GetArray(rqa_item[0], feature, reduce_dimension)
                file_nested_array = np.compress([True if i % frame_reduction == 0 else False for i in range(file_nested_array.shape[0])], file_nested_array, axis=0)
                cost_matrix, wp = librosa.sequence.dtw(X=file_array.T, 
                                                    Y=file_nested_array[int(rqa_item[1][0][0]):int(rqa_item[1][-1][0])].T, 
                                                    metric='euclidean', 
                                                    weights_mul=np.array([np.sqrt([2]),1,1], 
                                                    dtype=np.float64))    #cosine rychlejsie
                dtw_distance = cost_matrix[wp[-1, 0], wp[-1, 1]]
                print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                    parsed_file[0], "0" if test[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(test[0]),
                    parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3], idx_nested + 1, len(rqa_list),
                    dtw_distance))
                if dtw_distance < hit_threshold:
                    have_hit = True
                    f = open("evalRQA_DTW_{}.txt".format(feature), "a")
                    result_list.append("{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1", dtw_distance))  # train[1] == label
                    f.write("{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1", dtw_distance))
                    f.close()
                    break
                else:
                    dist_list.append(dtw_distance)
                if nested_loop_count > 50:
                    break

            if have_hit == False:
                f = open("evalRQA_DTW_{}.txt".format(feature), "a")
                result_list.append("{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", min(dist_list)))  # train[1] == label
                f.write("{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", min(dist_list)))
                f.close()
            dtw_time.append(time.time() - start)
            print('Next File. Time: {}s'.format(dtw_time[-1]))
        print(np.sum(dtw_time), np.mean(dtw_time))
        ft = open("timeRQA_DTW_{}.txt".format(feature), "a")
        ft.write("RQA_MEAN:{}\tRQA_TOTAL:{}\tDTW_MEAN:{}\tDTW_TOTAL:{}\tTOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(
                                        np.mean(rqa_time), np.sum(rqa_time), 
                                        np.mean(dtw_time), np.sum(dtw_time), 
                                        np.mean(rqa_time)+np.mean(dtw_time), np.sum(rqa_time)+np.sum(dtw_time)))
        ft.close()
    return result_list