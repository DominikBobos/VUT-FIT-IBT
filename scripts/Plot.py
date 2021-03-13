import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import librosa
import librosa.display


phn_labels = ['a', 'a:', 'au', 'b', 'c', 'd', 'dZ', 'dz', 'e', 'e:', 
				'eu', 'F', 'f', 'g', 'h_', 'i', 'i:', 'int', 'J', 
				'J_', 'j', 'k', 'l', 'm', 'N', 'n', 'o', 'o:', 'ou', 
				'P_', 'p', 'pau', 'r', 'S', 's', 'spk', 't', 'tS', 
				'ts', 'u', 'u:', 'v', 'x', 'Z', 'z', 'oth']


def Plot(feature1=None, feature2=None, dist=None, wp=None, sim_list=None, dtw_name="", info=[], gram_matrix=None):
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
		ax.tick_params(axis='x')#, rotation=20)
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


def PlotPhnAudio(phn_posteriors=None, file=None, info=[]):
	feat_name = 'Features'
	filename = ''
	if info != []:
		# tick = info[0][-1] / phn_posteriors.shape[0] 
		if info[0][-2] == "lin":
			feat_name = 'Phoneme posteriors'
		else:
			feat_name = 'MFCC'
		if len(info[0]) < 11:
			filename = info[0][0]
		else:
			filename = info[0][0] + ' ' + info[0][3]
	
	tick = 0.01	
