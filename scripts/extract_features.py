import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import soundfile as sf
from scipy.io import wavfile
# from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from python_speech_features import mfcc, logfbank
import sys
from matplotlib import cm
import librosa
import librosa.display
from dtw import accelerated_dtw
from scipy.spatial.distance import cdist


def my_dtw(o, r):
    cost_matrix = cdist(o, r, metric='seuclidean')
    m, n = np.shape(cost_matrix)

    for i in range(m):
        for j in range(n):
            if ((i == 0) & (j == 0)):
                cost_matrix[i, j] = cost_matrix[i, j]  # inf
            elif (i == 0):
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i, j - 1]  # inf
            elif (j == 0):
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i - 1, j]  # inf
            else:
                min_local_dist = np.min([cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]])
                cost_matrix[i, j] = cost_matrix[i, j] + min_local_dist
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


# plt.show()


# EXTRACTING FEATUES


# frequency_sampling, audio_signal = wavfile.read(sys.argv[1])

# print(frequency_sampling)
# features_mfcc = mfcc(audio_signal, frequency_sampling, 0.025, 0.01, 13, 512)
"""
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])
print(features_mfcc)

fig = plt.figure(figsize=(14, 10))
fig.suptitle("FILE: " + sys.argv[1].split('/')[-1], fontsize=12)

ax = fig.add_subplot(2, 1, 1)
# fig, ax = plt.subplots()
ax.set_title('MFCC')
ax.set_xlabel("window")
ax.set_ylabel("MFCC")

features_mfcc = np.swapaxes(features_mfcc, 0 ,1)
cax = ax.imshow(features_mfcc, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')

# ax.plot(features_mfcc)
# ax = fig.add_subplot(2, 2, 2)
# ax.set_title('all MFCC in one window')
# ax.set_xlabel("niecoX")
# ax.set_ylabel("niecoY")
# ax.plot(features_mfcc.T)

filterbank_features = logfbank(audio_signal, frequency_sampling, 0.025, 0.01, 26, 512)

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])
print(filterbank_features)

# filterbank_features = filterbank_features.T
ax = fig.add_subplot(2,1,2)
ax.set_title('Filter bank')
ax.set_xlabel("window")
ax.set_ylabel("Filter bank")
# ax.plot(filterbank_features)
filterbank_features = np.swapaxes(filterbank_features, 0 ,1)
cax = ax.imshow(filterbank_features, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')


# ax = fig.add_subplot(2,2,4)
# ax.set_title('Filter bank in one window')
# ax.set_xlabel("niecoX")
# ax.set_ylabel("niecoY")
# ax.plot(filterbank_features.T)
plt.show()
"""

sample_f1, signal1 = wavfile.read("../mixed/sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).wav")
sample_f2, signal2 = wavfile.read("../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav")
# sample_f2, signal2 = wavfile.read("../mixed/sw03864-A_9_20__A02_ST(0.00)L(31.77)G(2.80)R(9.69)S(1.03).wav")
# sample_f2, signal2 = wavfile.read("../mixed/sw03035-B_5_20__A01_ST(0.00)L(21.96)G(4.89)R(7.72)S(1.08).wav")

mfcc1 = mfcc(signal1, sample_f1, 0.025, 0.01, 13, 512)
mfcc2 = mfcc(signal2, sample_f2, 0.025, 0.01, 13, 512)
# print(sample_f1, sample_f2)
# Loading audio files
# y1, sr1 = librosa.load(sys.argv[1]) 
# y2, sr2 = librosa.load("../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav")


# from HTK import HTKFile


# htk_reader_1 = HTKFile()
# htk_reader_1.load("/home/dominik/Desktop/bak/short1.lin")
# print(htk_reader_1.nSamples, htk_reader_1.nFeatures, htk_reader_1.sampPeriod, htk_reader_1.qualifiers)
# mfcc1 = np.array(htk_reader_1.data)
# print(mfcc1.T.shape, mfcc1)

# htk_reader_2 = HTKFile()
# htk_reader_2.load("/home/dominik/Desktop/bak/short2.lin")
# print(htk_reader_2.nSamples, htk_reader_2.nFeatures, htk_reader_2.sampPeriod, htk_reader_2.qualifiers)
# mfcc2 = np.array(htk_reader_2.data)
# print(mfcc2.T.shape, mfcc2)



fig = plt.figure(figsize=(8, 6))
# fig.suptitle("FILE: " + sys.argv[1].split('/')[-1], fontsize=12)
ax = fig.add_subplot(1, 2, 1)
# fig, ax = plt.subplots()
ax.set_title('File 1')
ax.set_xlabel("Time [s]")
ax.set_ylabel("MFCC")

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
ax.xaxis.set_major_formatter(ticks_x)
# ax.xaxis.set_ticks(np.arange(0,mfcc1.shape[0],1))
features_mfcc1 = np.swapaxes(mfcc1, 0, 1)
cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto',
                label="energy")

ax = fig.add_subplot(1, 2, 2)
ax.set_title('File 2')
ax.set_xlabel("Time [s]")
ax.set_ylabel("MFCC")
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
ax.xaxis.set_major_formatter(ticks_x)
features_mfcc2 = np.swapaxes(mfcc2, 0, 1)
cax = ax.imshow(features_mfcc2, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')

# #Showing multiple plots using subplot
# plt.subplot(1, 2, 1)
# # mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
# librosa.display.specshow(mfcc1.T)
# plt.title('FILE1 MFCC')
# plt.xlabel('Time')
# plt.ylabel('MFCC')
# plt.subplot(1, 2, 2)
# # mfcc2 = librosa.feature.mfcc(y2, sr2)
# librosa.display.specshow(mfcc2.T)
# plt.title('FILE2 MFCC')
# plt.xlabel('Time')
# plt.ylabel('MFCC')

# dist, cost_matrix, acc_cost_matrix, path  = accelerated_dtw(mfcc1.T, mfcc2.T,"canberra")
# print("The normalized distance between the two : ",dist)   # 0 for similar audios

# plt.imshow(cost_matrix.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
# plt.plot(path[0], path[1], 'w')   #creating plot for DTW

# plt.show()  #To display the plots graphically




# dist, cost_matrix, acc_cost_matrix, path  = accelerated_dtw(mfcc1.T, mfcc2.T,"canberra")
dist, wp = librosa.sequence.dtw(X=mfcc1.T, Y=mfcc2.T, metric='seuclidean')
test = my_dtw(mfcc1, mfcc2)
print("My distance", test[1])
print("Distance", dist[wp[-1, 0], wp[-1, 1]])
# dist, cost_matrix, acc_cost_matrix, path  = accelerated_dtw(mfcc1.T, mfcc2.T,"canberra")

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(2,1,1)
# plt.subplot(2, 1, 1)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
ax.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * 0.01))
ax.yaxis.set_major_formatter(ticks_y)
# librosa.display.specshow(dist)
cax = ax.imshow(dist, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='auto')
ax.set_xlabel("File 2 Time [s]")
ax.set_ylabel("File 1 Time [s]")
ax.set_title('Librosa DTW alignment path')
ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='r')
ax.legend()


ax = fig.add_subplot(2,1,2)
# plt.subplot(2, 1, 1)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
ax.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * 0.01))
ax.yaxis.set_major_formatter(ticks_y)
# librosa.display.specshow(dist)
cax = ax.imshow(test[0], interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='auto')
ax.set_xlabel("File 2 Time [s]")
ax.set_ylabel("File 1 Time [s]")
ax.set_title('My DTW alignment path')
ax.plot(test[2][:, 1], test[2][:, 0], label='Optimal path', color='r')
ax.legend()

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# librosa.display.specshow(dist, x_axis='time', y_axis='time',
#                          cmap='gray_r', hop_length=mfcc2.shape[0])
# imax = ax.imshow(dist, cmap=plt.get_cmap('gray_r'),
#                  origin='lower', interpolation='nearest', aspect='auto')
# ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
# plt.title('Warping Path on Acc. Cost Matrix $D$')
# plt.colorbar()


# fig = plt.figure(figsize=(16, 8))
#
# # Plot x_1
# plt.subplot(2, 1, 1)
# librosa.display.waveplot(signal1, sr=sample_f1)
# plt.title('Slower Version $X_1$')
# ax1 = plt.gca()
#
# # Plot x_2
# plt.subplot(2, 1, 2)
# librosa.display.waveplot(signal2, sr=sample_f2)
# plt.title('Slower Version $X_2$')
# ax2 = plt.gca()
#
# plt.tight_layout()
#
# trans_figure = fig.transFigure.inverted()
# lines = []
# arrows = 30
# points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))
#
# # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
# for tp1, tp2 in wp[points_idx]:
#     # get position on axis for a given index-pair
#     coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
#     coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))
#
#     # draw a line
#     line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
#                                    (coord1[1], coord2[1]),
#                                    transform=fig.transFigure,
#                                    color='r')
#     lines.append(line)
#
# fig.lines = lines
# plt.tight_layout()

plt.tight_layout()
plt.show()
