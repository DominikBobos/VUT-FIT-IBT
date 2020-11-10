import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import sys
from matplotlib import cm
import librosa
import librosa.display
from scipy.spatial.distance import cdist
import pyaudio
import wave



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


def load_HTK(file1, file2):
	from HTK import HTKFile
	htk_reader_1 = HTKFile()
	htk_reader_1.load(file1)
	feature1 = np.array(htk_reader_1.data)

	htk_reader_2 = HTKFile()
	htk_reader_2.load(file2)
	feature2 = np.array(htk_reader_2.data)

	print("FILE | SAMPLES NUM | FEATURES NUM | SAMPLE PERIOD | QUALIFIERS")
	print(file1, htk_reader_1.nSamples, htk_reader_1.nFeatures, htk_reader_1.sampPeriod, htk_reader_1.qualifiers, sep=" | ")
	print(file2, htk_reader_2.nSamples, htk_reader_2.nFeatures, htk_reader_2.sampPeriod, htk_reader_2.qualifiers, sep=" | ")
	print("FILE | NUMPY ARRAY SHAPE")
	print(file1, feature1.shape, sep=" | ")
	print(file2, feature2.shape, sep=" | ")
	return feature1, feature2


def get_MFCC(file1, file2):
	sample_f1, signal1 = wavfile.read(file1)
	sample_f2, signal2 = wavfile.read(file2)

	mfcc1 = mfcc(signal1, sample_f1, 0.025, 0.01, 13, 512)
	mfcc2 = mfcc(signal2, sample_f2, 0.025, 0.01, 13, 512)

	print("FILE | SAMPLE | NUMPY ARRAY SHAPE")
	print(file1, sample_f1, mfcc1.shape, sep=" | ")
	print(file2, sample_f2, mfcc2.shape, sep=" | ")
	return mfcc1, mfcc2


def plot(feature1=None, feature2=None, dist=None, wp=None, sim_list=None, feat_name="Feature", dtw_name=""):

	if feature1 is not None and feature2 is not None:
		fig = plt.figure(figsize=(8, 6))
		# fig.suptitle("FILE: " + sys.argv[1].split('/')[-1], fontsize=12)
		ax = fig.add_subplot(1, 2, 1)
		# fig, ax = plt.subplots()
		ax.set_title('File 1')
		ax.set_xlabel("Time [s]")
		ax.set_ylabel(feat_name)

		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
		ax.xaxis.set_major_formatter(ticks_x)
		# ax.xaxis.set_ticks(np.arange(0,mfcc1.shape[0],1))
		features_mfcc1 = np.swapaxes(feature1, 0, 1)
		cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto',
						label="energy")

		ax = fig.add_subplot(1, 2, 2)
		ax.set_title('File 2')
		ax.set_xlabel("Time [s]")
		ax.set_ylabel(feat_name)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
		ax.xaxis.set_major_formatter(ticks_x)
		features_mfcc2 = np.swapaxes(feature2, 0, 1)
		cax = ax.imshow(features_mfcc2, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')

	if dist is not None and wp is not None:
		fig = plt.figure(figsize=(6, 4))
		ax = fig.add_subplot(1,1,1)
		# plt.subplot(2, 1, 1)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 0.01))
		ax.xaxis.set_major_formatter(ticks_x)
		ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * 0.01))
		ax.yaxis.set_major_formatter(ticks_y)
		# librosa.display.specshow(dist)
		cax = ax.imshow(dist, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='auto')
		ax.set_xlabel("File 2 Time [s]")
		ax.set_ylabel("File 1 Time [s]")
		ax.set_title(dtw_name + ' DTW alignment path')
		ax.plot(wp[:, 1], wp[:, 0], label='Optimal path',  color='coral', linewidth=2.0)
		if sim_list is not None:
			color=iter(cm.rainbow(np.linspace(0,1,len(sim_list))))
			for arr in sim_list:
				c = next(color)
				ax.plot(arr[:, 1], arr[:, 0], label='similarity',  color=c, linewidth=4.0)#, linestyle='dotted')
		ax.legend()
	
	if (feature1 is not None and feature2 is not None) or (dist is not None and wp is not None):
		plt.tight_layout()


def similarity(wp):
	sim_list = []
	tmp_list = []
	false_trend = 0
	constant_false = 0
	for point in zip(*[iter(np.flip(wp,0))]*2):
		# print(point)
		if point[0][0] < point[1][0] and point[0][1] < point[1][1]:
			tmp_list.append(point[0])
			tmp_list.append(point[1])
			constant_false = 0
		elif len(tmp_list) < 4:
			tmp_list = []
			false_trend = 0
			constant_false = 0
		else:
			tmp_list.append(point[0])
			tmp_list.append(point[1])
			false_trend += 1
			constant_false += 1

		if (false_trend > 0 and len(tmp_list)/false_trend < 4) or constant_false > 4:
			for i in range(constant_false):
				if len(tmp_list) > 0: 
					del tmp_list[-1]
			if ((false_trend - constant_false) > 0 and \
				len(tmp_list)/(false_trend - constant_false) > 4 and \
				len(tmp_list)) > 150 or \
				((false_trend - constant_false) == 0 and len(tmp_list) > 150):
				sim_list.append(np.array(tmp_list))
			tmp_list = []
			false_trend = 0
			constant_false = 0
		if len(tmp_list) > 400:
			for i in range(constant_false):
				if len(tmp_list) > 0: 
					del tmp_list[-1]
			sim_list.append(np.array(tmp_list))
			tmp_list = []
			false_trend = 0
			constant_false = 0

	if len(tmp_list) > 150:
		sim_list.append(np.array(tmp_list, dtype=object))
	sim_list = np.array(sim_list, dtype=object)
	return sim_list


def playback(files, sim_list):
	# set desired values

	for similarity in sim_list:
		for idx, file in enumerate(files):
			start = similarity[0, idx] /100
			length = (similarity[-1, idx] - similarity[0, idx])/100
			# open wave file
			wave_file = wave.open(file, 'rb')

			# initialize audio
			py_audio = pyaudio.PyAudio()
			stream = py_audio.open(format=py_audio.get_format_from_width(wave_file.getsampwidth()),
								   channels=wave_file.getnchannels(),
								   rate=wave_file.getframerate(),
								   output=True)

			# skip unwanted frames
			n_frames = int(start * wave_file.getframerate())
			wave_file.setpos(n_frames)

			# write desired frames to audio buffer
			n_frames = int(length * wave_file.getframerate())
			frames = wave_file.readframes(n_frames)
			stream.write(frames)

			# close and terminate everything properly
			stream.close()
			py_audio.terminate()
			wave_file.close()

			import time
			sys.stdout.write('\r\a{i}'.format(i=1))
			sys.stdout.flush()
			time.sleep(1)
			sys.stdout.write('\n')



file1 = "/home/dominik/Desktop/bak/short1.lin"
file2 = "/home/dominik/Desktop/bak/short2.lin"
# file2 = sw02220-A_0_180__A02_ST(0.00)L(46.49)G(1.74)R(13.53)S(0.98).lin

# file1 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).wav"
# file2 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav"
# file2 = "../../../mixed/sw03864-A_9_20__A02_ST(0.00)L(31.77)G(2.80)R(9.69)S(1.03).wav"
# file2 = "../../../mixed/sw03035-B_5_20__A01_ST(0.00)L(21.96)G(4.89)R(7.72)S(1.08).wav"

# feature1, feature2 = get_MFCC(file1, file2)
feature1, feature2 = load_HTK(file1, file2)
cost_matrix1, wp1 = librosa.sequence.dtw(X=feature1.T, Y=feature2.T, metric='seuclidean')
# cost_matrix2, dist2, wp2 = my_dtw(feature1, feature2)
# print("My distance", dist2)
print("Distance", cost_matrix1[wp1[-1, 0], wp1[-1, 1]])
sim_list1 = similarity(wp1)
# sim_list2 = similarity(wp2)
plot(feature1, feature2, cost_matrix1, wp1, sim_list1, feat_name="posteriors", dtw_name="Librosa")
# plot(dist=cost_matrix2, wp=wp2, sim_list=sim_list2, dtw_name="My")
plt.show()

playback([file1, file2], sim_list1)



