import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import wavfile
from scipy import ndimage
from python_speech_features import mfcc, logfbank
import sys
from matplotlib import cm
import librosa
import librosa.display
from scipy.spatial.distance import cdist
import pyaudio
import wave


phn_labels = ['a', 'a:', 'au', 'b', 'c', 'd', 'dZ', 'dz', 'e', 'e:', 
				'eu', 'F', 'f', 'g', 'h_', 'i', 'i:', 'int', 'J', 
				'J_', 'j', 'k', 'l', 'm', 'N', 'n', 'o', 'o:', 'ou', 
				'P_', 'p', 'pau', 'r', 'S', 's', 'spk', 't', 'tS', 
				'ts', 'u', 'u:', 'v', 'x', 'Z', 'z', 'oth']


mat10x10 = np.array([[-9, -8, -7, -6, -5, -4, -3, -2, -1, 0], 
					[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1], 
					[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
					[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3],
					[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
					[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
					[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
					[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
					[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
					[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

mat10x10N = np.array([[-9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 
					[8, 7, 6, 5, 4, 3, 2, 1, 0, -1], 
					[7, 6, 5, 4, 3, 2, 1, 0, -1, -2],
					[6, 5, 4, 3, 2, 1, 0, -1, -2, -3],
					[5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
					[4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
					[3, 2, 1, 0, -1, -2, -3, -4, -5, -6],
					[2, 1, 0, -1, -2, -3, -4, -5, -6, -7],
					[1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
					[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]])


mat10x10T = np.array([[0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
						[1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
						[2, 1, 0, -1, -2, -3, -4, -5, -6, -7],
						[3, 2, 1, 0, -1, -2, -3, -4, -5, -6],
						[4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
						[5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
						[6, 5, 4, 3, 2, 1, 0, -1, -2, -3],
						[7, 6, 5, 4, 3, 2, 1, 0, -1, -2],
						[8, 7, 6, 5, 4, 3, 2, 1, 0, -1],
						[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]) 


diag_matrix= np.zeros((100, 100), np.int)
# np.fill_diagonal(np.fliplr(a), 1)  # flip
np.fill_diagonal(diag_matrix, 1)  


mat5x5 = np.array([	[4, 3, 2, 1, 0], 
					[3, 2, 1, 0, -1], 
					[2, 1, 0, -1, -2],
					[1, 0, -1, -2, -3],
					[0, -1, -2, -3, -4]])


mat5x5N = np.array([[-4, -3, -2, -1, 0], 
					[-3, -2, -1, 0, 1], 
					[-2, -1, 0, 1, 2],
					[-1, 0, 1, 2, 3],
					[0, 1, 2, 3, 4]])

mat3x3 = np.array([	[2, 1, 0], 
					[1, 0, -1], 
					[0, -1, -2]])

mat3x3N = np.array([[-2, -1, 0], 
					[-1, 0, 1], 
					[0, 1, 2]])

mat3x3T = np.array([[0, -1, -2], 
					[1, 0, -1], 
					[2, 1, 0]])


def my_dtw(o, r):
	cost_matrix = cdist(o, r, metric='euclidean')
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
				min_cost = np.min([cost_matrix[i, j] + cost_matrix[i - 1, j],
							cost_matrix[i, j] + cost_matrix[i, j - 1],
							cost_matrix[i, j]*np.sqrt(2) + cost_matrix[i - 1, j - 1]])
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


def plot(feature1=None, feature2=None, dist=None, wp=None, sim_list=None, dtw_name="", info=[], gram_matrix=None):
	feat_name = 'Features'
	filename1 = ''
	filename2 = ''
	if info != []:
		tick1 = info[0][-1] / feature1.shape[0] 
		tick2 = info[1][-1] / feature2.shape[0]
		if info[0][-2] == "lin":
			feat_name = 'Phoneme posteriors'
		else:
			feat_name = 'MFCC'
		if len(info[0]) < 11:
			filename1 = info[0][0]
		else:
			filename1 = info[0][0] + ' ' + info[0][3]
		if len(info[0]) < 11:
			filename2 = info[1][0]
		else:
			filename2 = info[1][0] + ' ' + info[1][3]
	else:
		tick1 = 0.01
		tick2 = 0.01	

	if feature1 is not None and feature2 is not None:
		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(1, 2, 1)
		ax.set_title('File 1: {}'.format(filename1))
		ax.set_xlabel("Time [s]")
		ax.set_ylabel(feat_name)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick1))
		ax.xaxis.set_major_formatter(ticks_x)
		ax.tick_params(axis='x', rotation=20)
		if (feature1.shape[1] == 46):
			plt.yticks(np.arange(len(phn_labels)), phn_labels)
		features_mfcc1 = np.swapaxes(feature1, 0, 1)
		cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
		fig.colorbar(cax, ax=ax)

		# ax = fig.add_subplot(2, 2, 3)
		# signal = np.fromstring(wave.open(file1.replace('lin', 'wav'), "r").readframes(-1), "Int16")
		# ax.plot(signal)

		ax = fig.add_subplot(1, 2, 2)
		ax.set_title('File 2: {}'.format(filename2))
		ax.set_xlabel("Time [s]")
		# ax.set_ylabel(feat_name)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick2))
		ax.xaxis.set_major_formatter(ticks_x)
		ax.tick_params(axis='x', rotation=20)
		if (feature2.shape[1] == 46):
			plt.yticks(np.arange(len(phn_labels)), phn_labels)
		features_mfcc2 = np.swapaxes(feature2, 0, 1)
		cax = ax.imshow(features_mfcc2, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
		fig.colorbar(cax, ax=ax, label="probability")
		
		# ax = fig.add_subplot(2, 2, 4)
		# signal = np.fromstring(wave.open(file2.replace('lin', 'wav'), "r").readframes(-1), "Int16")
		# ax.plot(signal)

	if dist is not None and wp is not None:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick2))
		ax.xaxis.set_major_formatter(ticks_x)
		ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * tick1))
		ax.yaxis.set_major_formatter(ticks_y)
		ax.tick_params(axis='x', rotation=20)
		if gram_matrix is not None:
			cax = ax.imshow(gram_matrix, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
			fig.colorbar(cax, ax=ax, label="Gram matrix values")
		else:
			cax = ax.imshow(dist, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
			fig.colorbar(cax, ax=ax, label="distance cost")
		ax.set_xlabel("File 2 Time [s]")
		ax.set_ylabel("File 1 Time [s]")
		fig.suptitle(dtw_name + ' DTW alignment path')
		ax.set_title("Distance: {0:.6f}".format(dist[wp[-1, 0], wp[-1, 1]]))
		
		ax.plot(wp[:, 1], wp[:, 0], label='Optimal path',  color='coral', linewidth=2.0)
		if sim_list is not None:
			color=iter(cm.rainbow(np.linspace(0,1,len(sim_list))))
			for arr in sim_list:
				c = next(color)
				ax.plot(arr[:, 1], arr[:, 0], label='similarity',  color=c, linewidth=4.0)#, linestyle='dotted')
		ax.legend()
	plt.tight_layout()
	# plt.savefig('DTW.png')
	if (feature1 is not None and feature2 is not None) or (dist is not None and wp is not None):
		pass


def plot_phn_audio(phn_posteriors=None, file=None ,info=[]):
	feat_name = 'Features'
	filename = ''
	if info != []:
		tick = info[0][-1] / phn_posteriors.shape[0] 
		if info[0][-2] == "lin":
			feat_name = 'Phoneme posteriors'
		else:
			feat_name = 'MFCC'
		if len(info[0]) < 11:
			filename = info[0][0]
		else:
			filename = info[0][0] + ' ' + info[0][3]
	else:
		tick = 0.01	

	if phn_posteriors is not None:
		fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
		# fig = plt.figure(figsize=(8, 6))
		# ax = fig.add_subplot(2, 1, 1)
		ax.set_title('File 1: {}, repeat: {} times,'.format(filename, info[0][7]))
		ax.set_xlabel("Time [s]")
		ax.set_ylabel(feat_name)
		ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick))
		ax.xaxis.set_major_formatter(ticks_x)
		ax.tick_params(axis='x', rotation=20)
		if (phn_posteriors.shape[1] == 46):
			ax.yaxis.set_major_locator(plt.MaxNLocator(46))
			ax.yaxis.set_major_formatter(ticker.IndexFormatter(phn_labels))
			# labels = [item.get_text() for item in ax.get_yticklabels()]
			# labels[1] = 'Testing'
			# ax.set_yticklabels(labels)
			# plt.yticks(np.arange(len(phn_labels)), phn_labels)
		features_mfcc1 = np.swapaxes(phn_posteriors, 0, 1)
		cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
		# ax2 = fig.add_subplot(2, 1, 2)
		spf = wave.open(file.replace("lin", "wav"), "r")
		# Extract Raw Audio from Wav File
		signal = spf.readframes(-1)
		signal = np.frombuffer(signal, dtype='int16')
		fs = spf.getframerate()
		time = np.linspace(0, len(signal) / fs, num=len(signal))
		# ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick/100))
		# ax2.xaxis.set_major_formatter(ticks_x)
		# ax2.tick_params(axis='x', rotation=20)
		ax2.plot(time, signal)
		plt.axis(xmin=0,xmax=time[-1])


def similarity(wp):
	sim_list = []
	tmp_list = []
	false_trend = 0
	constant_false = 0
	for point in zip(*[iter(np.flip(wp,0))]*2):	#take two in a row
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
			if (false_trend != 0 and \
				len(tmp_list)/false_trend > 5) or \
				false_trend == 0:
				sim_list.append(np.array(tmp_list))

			tmp_list = []
			false_trend = 0
			constant_false = 0

	if len(tmp_list) >= 100:
		sim_list.append(np.array(tmp_list))
	for i in range(len(sim_list)-1,-1,-1):
		if len(sim_list[i]) < 100:
			sim_list.pop(i)
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



def reduce_dimension(feature):
	reduced = np.empty([feature.shape[0], feature.shape[1] // 3])

	for frame in range(feature.shape[0]):
		for i in range(0, feature.shape[1], 3):
			reduced[frame, i//3] = (feature[frame, i] + feature[frame, i+1] + feature[frame, i+2])
	print(reduced.shape)
	return reduced


def parse(filename):
	parsed_file = []
	del_path = ''
	temp = ''
	try:
		del_path = filename.split('/')[-1]
	except IndexError:
		del_path = filename
	temp = del_path.split('_')
	parsed_file.append(temp[0])	 # sample ID name
	parsed_file.append(int(temp[1]))  # part of cut sample
	# <20 -> sample without message 
	if len(del_path) < 20:
		parsed_file.append(float(temp[2].split('.')[0]))	# sample duration in seconds
		parsed_file.append(temp[2].split('.')[-1])		# extension format (wav, lin etc.)
		parsed_file.append(temp[2])	# total duration
		return parsed_file
	else:
		parsed_file.append(float(temp[2]))  # sample duration in seconds
	temp = del_path.split('__')[-1]		# part with operator message info
	parsed_file.append(temp.split('_')[0])	# operator message ID
	temp = temp.split('_')[-1].split('(')
	parsed_file.append(float(temp[1].split(')')[0]))	# message start point
	parsed_file.append(float(temp[2].split(')')[0]))	# message duration in seconds
	parsed_file.append(float(temp[3].split(')')[0]))	# volume gain
	parsed_file.append(float(temp[4].split(')')[0]))	# repeat count
	parsed_file.append(float(temp[5].split(')')[0]))	# speed
	parsed_file.append(temp[5].split('.')[-1])			# extension format (wav, lin etc.)
	parsed_file.append(parsed_file[2] + parsed_file[5])	# total duration of mixed file
	# [sampleID, cutPart, sampleDuration, messageID, messStart, messDuration, volume, repeat, speed, extension, totalDuration]
	return parsed_file	  	


def gram_matrix(feature):
	matrix = feature.dot(feature.T)#(feature.dot(feature.T)) / (np.linalg.norm(feature) * np.linalg.norm(feature.T))			
	return 0.5 * (matrix + 1)	
	# return feature


def image_filter(matrix, threshold=0.7, percentile=70, variance=5):
	matrix[matrix < threshold] = 0.0
	matrix[matrix >= threshold] = 1.0
	matrix = ndimage.percentile_filter(matrix, percentile=percentile, footprint=diag_matrix, mode='constant', cval=0.0)
	matrix = ndimage.gaussian_filter(matrix, variance)
	return matrix


# file1 = "/home/dominik/Desktop/bak/short1.lin"
# file2 = "/home/dominik/Desktop/bak/short2.lin"
# file1 = "/home/dominik/Desktop/bak/A00_3.60.lin"
# file2 = "/home/dominik/Desktop/bak/A00_6.36.lin"
# file2 = '../../../sw02220-A_0_180__A02_ST(0.00)L(46.49)G(1.74)R(13.53)S(0.98).lin'

file1 = "../../sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).lin"
file2 = "../../sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).lin"
# file2 = "../../sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).lin"	#with reduced for some reason no hits
# file2 = "../../sw00000-A_0_0__B10_ST(0.00)L(165.93)G(4.10)R(25.78)S(1.03).lin" #with seuclidean metrics it makes similarities hits!
# file2 = "../../sw00000-A_0_0__B03_ST(0.00)L(10.25)G(5.45)R(3.71)S(1.06).lin"	#with reduced euclidean metrics it makes similarities hits!
# file1 = "../../sw00000-A_0_0__A10_ST(0.00)L(53.39)G(2.30)R(8.83)S(1.04).lin"
# file2 = "../../sw03521-B_1_45__A10_ST(0.00)L(9.05)G(5.79)R(1.51)S(1.07).lin"
# file1 = "../../sw03720-B_5_30__A02_ST(0.00)L(7.36)G(-0.61)R(2.26)S(1.04).lin"
# file2 = "../../sw03720-B_5_30__A02_ST(0.00)L(7.36)G(-0.61)R(2.26)S(1.04).lin"


# file1 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).wav"
# file2 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(10.06)G(0.18)R(2.88)S(0.95).wav"
# file1 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav"
# file2 = "../../../mixed/sw00000-A_0_0__A02_ST(0.00)L(44.80)G(5.09)R(14.45)S(1.09).wav"
# file2 = "../../../mixed/sw00000-A_0_0__B10_ST(0.00)L(165.93)G(4.10)R(25.78)S(1.03).wav" #with seuclidean metrics it makes similarities hits!
# file2 = "../../../mixed/sw00000-A_0_0__B03_ST(0.00)L(10.25)G(5.45)R(3.71)S(1.06).wav"	#with seuclidean metrics it makes similarities hits!
# file2 = "../../../mixed/sw03864-A_9_20__A02_ST(0.00)L(31.77)G(2.80)R(9.69)S(1.03).wav"
# file1 = "../../../mixed/sw03035-B_5_20__A01_ST(0.00)L(21.96)G(4.89)R(7.72)S(1.08).wav"
# file2 = "../../../mixed/sw03035-B_5_20__A01_ST(0.00)L(21.96)G(4.89)R(7.72)S(1.08).wav"



parsed1 = parse(file1)
parsed2 = parse(file2)

# feature1, feature2 = get_MFCC(file1, file2)
feature1, feature2 = load_HTK(file1, file2)
feature1 = reduce_dimension(feature1)
feature2 = reduce_dimension(feature2)

# import time
# count = []
# for i in range(10):
# 	start = time.time()
# 	cost_matrix1, wp1 = librosa.sequence.dtw(X=feature1.T, Y=feature2.T, metric='euclidean')	#cosine rychlejsie
# 	count.append(time.time()-start)
# print("euclidean", np.mean(count))

cost_matrix1, wp1 = librosa.sequence.dtw(X=feature1.T, Y=feature2.T, metric='euclidean', weights_mul=np.array([np.sqrt([2]),1,1], dtype=np.float))	#cosine rychlejsie
# cost_matrix2, dist2, wp2 = my_dtw(feature1, feature2)
# print(feature1)
print("Distance", cost_matrix1[wp1[-1, 0], wp1[-1, 1]])
# print("My distance", dist2)

sim_list1 = similarity(wp1)
# sim_list2 = similarity(wp2)


gram_matrix = gram_matrix(feature1)
gram_matrix = image_filter(gram_matrix)
# gram_matrix = None

plot(feature1, feature2, cost_matrix1, wp1, sim_list1, dtw_name="Librosa", info=[parsed1, parsed2], gram_matrix=gram_matrix)
# plot(dist=cost_matrix2, wp=wp2, sim_list=sim_list2, dtw_name="My", gram_matrix=None)
# plot_phn_audio(feature2, file=file2, info=[parsed2])

plt.show()



'''

#DETECT 45 DEGREE LINES (IMAGE DETECTION)

wp = np.flip(wp1,0)
data = np.zeros((wp[-1][0]+1, wp[-1][1]+1, 3))
for x in np.flip(wp1,0):
	data[x[0], x[1]] = [255, 255, 255]
plt.imshow(data, interpolation='bilinear',aspect='equal')
plt.savefig("Lines.png")
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Lines.png')

# plt.imshow(img)
# plt.show()

kernel = np.array([[0, -25, 1],
                   [-25, 5, -25],
                   [1, -25, 0]])

dst = cv2.filter2D(img, -1, kernel)
cv2.imwrite("filtered.png", dst)

'''



#PLAYBACK SIMILARITIES

# file1 = "../../../sw00000-A_0_0__A00_ST(0.00)L(3.60)G(4.89)R(1.73)S(1.08).wav"
# file2 = "../../../sw00000-A_0_0__A00_ST(0.00)L(6.36)G(2.21)R(2.91)S(0.99).wav"
# playback([file1, file2], sim_list1)



