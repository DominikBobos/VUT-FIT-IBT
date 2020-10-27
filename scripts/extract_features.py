import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
# import soundfile as sf
from scipy.io import wavfile
# from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from python_speech_features import mfcc, logfbank
import sys
from matplotlib import cm
import librosa
import librosa.display
from dtw import accelerated_dtw
# pip3 install python_speech_features


# nahravka = sys.argv[1]		#zmente pre ziadany subor
# #cesta = os.path.join(sentences, nahravka)
# s, fs = sf.read(nahravka)
# s = s[:250000]
# t = np.arange(s.size) / fs


#  #	segmenty o dlzke 25ms, prekrytie 15ms, 'N->512, lenze indexujeme od 0'

# f, t, sgr = spectrogram(s, fs,'25' ,400, 240, 511)
# sgr_log = 10 * np.log10(sgr+1e-20) 


# fig = plt.figure(figsize=(9,3))
# plt.pcolormesh(t,f,sgr_log)
# plt.gca().set_xlabel('Čas [s]')
# plt.gca().set_ylabel('Frekvencia [Hz]')
# plt.title(nahravka)
# ax1 = fig.add_subplot(111)
# plt.tight_layout()
# pocetvzoriek = sgr.shape
# print(pocetvzoriek[1])


# plt.show()


#EXTRACTING FEATUES



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

mfcc1 = mfcc(signal1, sample_f1, 0.025, 0.01, 13, 512)
mfcc2 = mfcc(signal2, sample_f2, 0.025, 0.01, 13, 512)
print(sample_f1, sample_f2)
#Loading audio files
# y1, sr1 = librosa.load(sys.argv[1]) 
# y2, sr2 = librosa.load("../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav")


fig = plt.figure(figsize=(4, 6))
# fig.suptitle("FILE: " + sys.argv[1].split('/')[-1], fontsize=12)
ax = fig.add_subplot(1, 2, 1)
# fig, ax = plt.subplots()
ax.set_title('MFCC')
ax.set_xlabel("Time [s]")
ax.set_ylabel("MFCC")
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*0.01))
ax.xaxis.set_major_formatter(ticks_x)
# ax.xaxis.set_ticks(np.arange(0,mfcc1.shape[0],1))
features_mfcc1 = np.swapaxes(mfcc1, 0 ,1)
cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')


ax = fig.add_subplot(1, 2, 2)
ax.set_title('MFCC')
ax.set_xlabel("window")
ax.set_ylabel("MFCC")
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*0.01))
ax.xaxis.set_major_formatter(ticks_x)
features_mfcc2 = np.swapaxes(mfcc2, 0 ,1)
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
dist, wp = librosa.sequence.dtw(X=mfcc1.T, Y=mfcc2.T, metric='cosine')
print(mfcc1.shape[0]*(0.01))
wp_s = np.asarray(wp) * mfcc2.shape[0]*(0.01)

# dist, cost_matrix, acc_cost_matrix, path  = accelerated_dtw(mfcc1.T, mfcc2.T,"canberra")

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
librosa.display.specshow(dist, x_axis='time', y_axis='time',
                         cmap='gray_r', hop_length=mfcc2.shape[0]*(0.01))
imax = ax.imshow(dist, cmap=plt.get_cmap('gray_r'),
                 origin='lower', interpolation='nearest', aspect='auto')
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
plt.title('Warping Path on Acc. Cost Matrix $D$')
plt.colorbar()
plt.show() 