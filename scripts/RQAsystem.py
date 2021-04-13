import sys
import librosa
import numpy as np
import time

import ArrayFromFeatures
import DTWsystem
import SpeechDetection
import Playback



def GetFeature(file):
    if file[-3:] == 'wav':
        return 'mfcc'
    elif file[-3:] == 'lin':
        return 'poteriors'
    elif file[-3:] == 'fea':
        return 'bottleneck'
    elif file[-3:] == 'str':
        return 'string'
    elif file[-4:] == 'latt':
        return 'lattice'
    else:
        raise TypeError("Unknown feature extension: {}".format(file[-3:]))


def CreateKnownCluster(files, feature=None, frame_reduction=1):
    try:
        if feature is None:
            feature = GetFeature(files[0])
        clust_list = ArrayFromFeatures.OpenPickle("evaluations/objects/cluster_known_list_{}.pkl".format(feature))
        print("Will Not perform clustering - clust_list loaded from existing pkl file.")
        return clust_list
    except:
        pass
    message_ids = []
    clust_list = []
    clust_time = []
    for idx, file in enumerate(files):
        start_time = time.time()
        if feature is not None:
            file = ArrayFromFeatures.RightExtensionFile(file, feature)
        parsed_file = ArrayFromFeatures.Parse(file)
        frames_start = int(parsed_file[4] * 100 // frame_reduction)
        frames_end = int((parsed_file[4] + parsed_file[5]) * 100 // frame_reduction)
        frames = [[frames_start, frames_start], [frames_end, frames_end]]
        duration = int(parsed_file[5] * 100) #conversion to hundredths of seconds
        if parsed_file[3] in message_ids:
            index = message_ids.index(parsed_file[3])
            # filepath, frames, frame_reduction, dtw_distance=-1, duration ->to use same format as unknown clusters
            clust_list[index].append([file, frames, frame_reduction, -1, duration])
            clust_list[index] = sorted(clust_list[index], key = lambda x: x[-1]) #sorted by length
        else:
            message_ids.append(parsed_file[3])
            # filepath, frames, frame_reduction, dtw_distance=-1, duration
            clust_list.append([[file, frames, frame_reduction, -1, duration]]) 
        clust_time.append(time.time() - start_time)
        print("Processing {}[{}] ({}/{}). Time: {}".format(
                parsed_file[0], parsed_file[3], idx + 1, len(files), clust_time[-1]))

    print("==============CLUSTERS==============")
    for idx, cluster in enumerate(clust_list):
        print("+----------------------------------+")
        print("Cluster:", idx)
        for item in cluster:
            print(item[0], item[1], item[-1])
        print("+__________________________________+")

    if feature is None:
        feature = GetFeature(files[0])
    ArrayFromFeatures.SavePickle("evaluations/objects/cluster_known_list_{}.pkl".format(feature), clust_list)
   
    ft = open("timeCluster_known_{}.txt".format(feature), "a")
    ft.write("TOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(np.mean(clust_time), np.sum(clust_time)))
    ft.close()

    return clust_list


def RqaCluster(rqa_list_pkl, feature, playback=False, metric='cosine', frame_reduction=None, rqa_list=None):
    try:
        clust_list = ArrayFromFeatures.OpenPickle("evaluations/objects/cluster_rqa_list_{}.pkl".format(feature))
        print("Will Not perform clustering - clust_list loaded from existing pkl file.")
        return clust_list
    except:
        pass
    hit_threshold = DTWsystem.GetThreshold('rqa_cluster', feature, metric)
    rqa_list = ArrayFromFeatures.OpenPickle(rqa_list_pkl) if rqa_list is None else rqa_list
    rqa_list_clear = []
    clust_time = []
    clust_list = []
    cache = {}
    if frame_reduction is None:
        frame_reduction = rqa_list[0][2]        # get the reduction from the imported rqa_list 
    for audio in rqa_list:
        if len(audio[1]) == 0:      # empty list (new Pythonic way cuz comparing to [] will be error)
            continue
        start, end, length = Playback.FramesToHSeconds(audio[1], audio[2])  
        if SpeechDetection.SpeechInAudio(audio[0], start/100, end/100): # if voiced part of audio 
            rqa_list_clear.append(audio)
            # playbacks miss 
            if playback and len(audio[0].split('/')[-1]) < 30: # non-target have filenames shorter than 30
                Playback.Playback(file=audio[0], sim_list=[start/100, end/100])
    # cluster by the closest distances
    for idx, file in enumerate(rqa_list_clear):
        # if file is already processed
        if file[0] in [item for sublist in clust_list for subsublist in sublist for item in subsublist]:
            continue
        start_time = time.time()
        new_cluster = []   # filepath, frames to compare, frame_reduction, DTWdistance, length of sliced audio 
        if file[0].split('/')[-1] in cache:
            parsed_file, file_array = cache[file[0].split('/')[-1]]
        else:
            parsed_file = ArrayFromFeatures.Parse(file[0])
            file_array = ArrayFromFeatures.GetArray(file[0], feature, True)
            file_array = ArrayFromFeatures.ReduceFrames(file_array, size=frame_reduction)
            cache[file[0].split('/')[-1]] = [parsed_file, file_array]
        new_cluster.append([file[0], file[1], file[2], 0.0, len(file[1])*file[2]])
        for idx_nested, file_nested in enumerate(rqa_list_clear):
            # if nested file is already processed
            if file_nested[0] in [item for sublist in clust_list for subsublist in sublist for item in subsublist]:
                continue
            if file[0].split('/')[-1] == file_nested[0].split('/')[-1]:
                continue
            if file_nested[0].split('/')[-1] in cache:
                parsed_file_nested, file_nested_array = cache[file_nested[0].split('/')[-1]]
            else:
                parsed_file_nested = ArrayFromFeatures.Parse(file_nested[0])
                file_nested_array = ArrayFromFeatures.GetArray(file_nested[0], feature, True)
                file_nested_array = ArrayFromFeatures.ReduceFrames(file_nested_array, size=frame_reduction)
                cache[file_nested[0].split('/')[-1]] = [parsed_file_nested, file_nested_array]
            # file_nested_array = ArrayFromFeatures.CompressFrames(file_nested_array, size=frame_reduction)
            start = int(file[1][0][0]) * file[2] // frame_reduction 
            end = int(file[1][-1][0]) * file[2] // frame_reduction
            start_nested = int(file_nested[1][0][0]) * file_nested[2] // frame_reduction 
            end_nested = int(file_nested[1][-1][0]) * file_nested[2] // frame_reduction
            path = DTWsystem.SegmentalDTW(file_nested_array[start_nested:end_nested], 
                                          file_array[start:end], 
                                          R=5, 
                                          L=200/frame_reduction, 
                                          dist=metric)
            # wp = np.asarray(path[1][3])*frame_reduction   # not necessary, uncomment when want to know the warping path
            dtw_distance = path[0]
            if feature == 'posteriors' and dtw_distance < 0.01:
                dtw_distance *= 100 # normalization for weird cases
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                parsed_file[0], "0" if len(parsed_file) < 10 else "1:" + parsed_file[3], idx + 1, len(rqa_list_clear),
                parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3],
                idx_nested + 1, len(rqa_list_clear), dtw_distance))
            if dtw_distance < hit_threshold:
                new_cluster.append([file_nested[0], file_nested[1], file_nested[2], 
                                    dtw_distance, len(file_nested[1])*file_nested[2]])
        clust_time.append(time.time() - start_time)
        print("Next file. Time:{}".format(clust_time[-1]))
        clust_list.append(sorted(new_cluster, key = lambda x: x[-1])) # sorted by lenth

    print("Processing clusters with only one item")
    cluster_with_one_file = []
    for idx, cluster in enumerate(clust_list):
        if len(cluster) == 1:
            cluster_with_one_file.append([idx, cluster[0]]) 
    for cluster in cluster_with_one_file:
        #chcem proste ocekovat ci sa nahodou clustre o velkosti jedna nedaju este niekde inde zaradit takze na ne zas pustim sdtw
        start_time = time.time()
        if cluster[1][0].split('/')[-1] in cache:
            parsed_file, file_array = cache[cluster[1][0].split('/')[-1]]
        else:
            parsed_file = ArrayFromFeatures.Parse(cluster[1][0])
            file_array = ArrayFromFeatures.GetArray(cluster[1][0], feature, True)
            file_array = ArrayFromFeatures.ReduceFrames(file_array, size=frame_reduction)
            cache[cluster[1][0].split('/')[-1]] = [parsed_file, file_array]
        for idx_nested, cluster_nested in enumerate(clust_list):
            have_hit = False
            if idx_nested in [index for sublist in cluster_with_one_file for index in sublist]: #comparing the cluster of size one 
                continue
            for idx_file, file in enumerate(cluster_nested):
                if idx_file > 2:   # go to next when first three are compared
                    break
                if file[0].split('/')[-1] in cache:
                    parsed_file_nested, file_nested_array = cache[file[0].split('/')[-1]]
                else:
                    parsed_file_nested = ArrayFromFeatures.Parse(file[0])
                    file_nested_array = ArrayFromFeatures.GetArray(file[0], feature, True)
                    file_nested_array = ArrayFromFeatures.ReduceFrames(file_nested_array, size=frame_reduction)
                    cache[file[0].split('/')[-1]] = [parsed_file_nested, file_nested_array]
                start = int(cluster[1][1][0][0]) * cluster[1][2] // frame_reduction 
                end = int(cluster[1][1][-1][0]) * cluster[1][2] // frame_reduction
                start_nested = int(file[1][0][0]) * file[2] // frame_reduction 
                end_nested = int(file[1][-1][0]) * file[2] // frame_reduction
                path = DTWsystem.SegmentalDTW(file_nested_array[start_nested:end_nested], 
                                              file_array[start:end], 
                                              R=5, 
                                              L=200/frame_reduction, 
                                              dist=metric)
                # wp = np.asarray(path[1][3])*frame_reduction   # not necessary, uncomment when want to know the warping path
                dtw_distance = path[0]
                if feature == 'posteriors' and dtw_distance < 0.01:
                    dtw_distance *= 100 # normalization for weird cases
                print("Refitting {}[{}] to existing cluster [{}] with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                    parsed_file[0], "0" if len(parsed_file) < 10 else "1:" + parsed_file[3], idx_nested,
                    parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3],
                    idx_file + 1, len(cluster_nested) if len(cluster_nested) < 3 else 3, dtw_distance))
                if dtw_distance < hit_threshold:
                    have_hit = True
                    clust_list[idx_nested].append([cluster[1][0], cluster[1][1], cluster[1][2], 
                                        dtw_distance, len(cluster[1][1])*cluster[1][2]])
                    clust_list[idx_nested] = sorted(clust_list[idx_nested], key = lambda x: x[-1]) # sort with new item
                    break
            if have_hit:
                break
        clust_time.append(time.time() - start_time)
    # remove cluster of size one from the cluster list
    for i in range(len(cluster_with_one_file)-1, -1, -1):
        del clust_list[cluster_with_one_file[i][0]]

    print("==============CLUSTERS==============")
    for idx, cluster in enumerate(clust_list):
        print("+----------------------------------+")
        print("Cluster:", idx)
        for item in cluster:
            print(item[0], item[-2])
        print("+__________________________________+")

    ArrayFromFeatures.SavePickle("evaluations/objects/cluster_rqa_list_{}_{}.pkl".format(feature, frame_reduction), clust_list)

    ft = open("timeClusterRQA_unknown_{}_{}.txt".format(feature,frame_reduction), "a")
    ft.write("TOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(np.mean(clust_time), np.sum(clust_time)))
    ft.close()

    return clust_list
    

def RqaAnalysis(data, rqa_threshold, frame_reduction, feature, reduce_dimension, second_pass):
    rqa_list = [] # [0]=filepath, [1]=frames, [2]=frame_reduction, [3]=score
    rqa_time = []
    result_list = []
    for idx, file in enumerate(data[0]):  # rqa analysis first
        start_time = time.time()
        parsed_file = ArrayFromFeatures.Parse(file)
        if float(parsed_file[-1]) < 3.0:  # shorter samples are skipped 
            rqa_list.append([file, [], frame_reduction, 10000.0])  # giving high score to that one
            print("Skipping", file.split('/')[-1], data[1][idx],
                  "because the total duration {} is less than 3.0 seconds".format(parsed_file[-1]),
                  time.time() - start_time)
            rqa_time.append(time.time() - start_time)
            if not second_pass:
                f = open("evalRQA_unknown_{}_{}.txt".format(feature, frame_reduction), "a")
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
        # reduced_array = ArrayFromFeatures.CompressFrames(file_array, frame_reduction)
        reduced_array = ArrayFromFeatures.ReduceFrames(file_array, frame_reduction)
        rec_matrix = librosa.segment.recurrence_matrix(reduced_array.T, width=40//frame_reduction, k=reduced_array.shape[0] // 10,
                                                       mode='affinity',
                                                       metric='cosine')  # ['connectivity', 'distance', 'affinity']
        score, path = librosa.sequence.rqa(rec_matrix, np.inf, np.inf, knight_moves=True)
        path_length = (int(path[-1][0]) - int(path[0][0])) * frame_reduction
        # only frames longer than 2.5 seconds are valid and giving high score penalty for them
        rqa_list.append([file, path, frame_reduction, np.sum(rec_matrix[path]) / (int(path[-1][0]) - int(path[0][0]))]
                        if path_length > 250 else [file, [], frame_reduction, 5000.0])
        rqa_time.append(time.time() - start_time)
        print("Processing {}[{}] ({}/{}), path length: {}, score: {}, time: {}".format(
            parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3],
            idx + 1, len(data[0]),
            path_length,
            rqa_list[-1][-1],
            time.time() - start_time))
        if not second_pass:
            f = open("evalRQA_unknown_{}_{}.txt".format(feature, frame_reduction), "a")
            append_string = "{}\t{}\t{}\t{}\t{}\n".format(
                file.split('/')[-1],
                data[1][idx],
                "0" if (rqa_list[-1][1] == []) or (rqa_list[-1][2] > rqa_threshold) else "1",
                path_length,
                rqa_list[-1][-1])
            result_list.append(append_string)
            f.write(append_string)
            f.close()
    print("RQA mean:", np.mean(rqa_time), "RQA total:", np.sum(rqa_time))
    if not second_pass:
        ft = open("timeRQA_unknown_{}_{}.txt".format(feature, frame_reduction), "a")
        ft.write("TOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(np.mean(rqa_time), np.sum(rqa_time)))
        ft.close()
    return rqa_list, rqa_time, result_list


def RqaDtw(train=None, test=None, feature='mfcc', frame_reduction=1, reduce_dimension=True, second_pass=False,
                  sdtw=False, cluster=False, metric='cosine', known=False):
    if frame_reduction < 1:
        raise ValueError("Frame reduction must be at least 1 or bigger")
    train, train_nested, test, test_nested = DTWsystem.InitCheckShuffle(train, test)
    result_list = []
    rqa_time = []
    rqa_list = []
    system = ''
    if not second_pass:
        system = 'arenjansen'
    elif not sdtw:
        system = '2pass_dtw_unknown'
    elif not cluster:
        system = '2pass_sdtw_unknown'
    if cluster:
        hit_threshold = DTWsystem.GetThreshold('rqa_cluster_system', feature, 'cosine')
    else:
        rqa_threshold, hit_threshold, loop_count = DTWsystem.GetThreshold(system, feature, metric)

    if not known:
        try:
            rqa_list = ArrayFromFeatures.OpenPickle("evaluations/objects/rqa_list_{}.pkl".format(feature))
        except:
            print("No rqa_list found, will tryy to create one")
    if not rqa_list and not known:
        rqa_list, rqa_time, result_list = RqaAnalysis(data=test, rqa_threshold=rqa_threshold,
                                                      frame_reduction=frame_reduction,
                                                      feature=feature, reduce_dimension=reduce_dimension,
                                                      second_pass=second_pass)
    if not rqa_time:
        rqa_time = [0.]

    if not known:
        ArrayFromFeatures.SavePickle("evaluations/objects/rqa_list_{}_{}.pkl".format(feature, frame_reduction), rqa_list)

    if second_pass and not cluster and not known:
        result_list = DTWsystem.SecondPassDtw(data=test, rqa_list=rqa_list, hit_threshold=hit_threshold,
                                              frame_reduction=frame_reduction, feature=feature,
                                              reduce_dimension=reduce_dimension,
                                              loop_count=loop_count, rqa_time=rqa_time, sdtw=sdtw, known=False)
    if second_pass and cluster and not known:
        clust_list = RqaCluster(None, feature, metric='cosine', frame_reduction=20, rqa_list=rqa_list)
        result_list = DTWsystem.SecondPassCluster(data=test, clust_list=clust_list, hit_threshold=hit_threshold, 
                                                  frame_reduction=frame_reduction, feature=feature, 
                                                  reduce_dimension=reduce_dimension, rqa_time=rqa_time, 
                                                  metric=metric, sdtw=sdtw, known=False)

    if second_pass and cluster and known:
        from main import GetFiles
        cluster_train = GetFiles('/home/dominik/Desktop/bak/dev_data/eval/eval_goal/', feature)
        clust_list = CreateKnownCluster(cluster_train, feature) # frame_reduction = frame reduction
        result_list = DTWsystem.SecondPassCluster(data=test, clust_list=clust_list, hit_threshold=hit_threshold, 
                                                  frame_reduction=frame_reduction, feature=feature, 
                                                  reduce_dimension=reduce_dimension, rqa_time=rqa_time, 
                                                  metric=metric, sdtw=sdtw, known=known)

    return result_list


if __name__ == "__main__":
    RqaCluster(sys.argv[1], sys.argv[2], metric=sys.argv[3], frame_reduction=int(sys.argv[4]), playback=False)