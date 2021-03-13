import numpy as np
from scipy.io import wavfile
from scipy import ndimage
from python_speech_features import mfcc, logfbank
from matplotlib import cm
import librosa
from scipy.spatial.distance import cdist


def MyDTW(o, r):
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


cost_matrix1, wp1 = librosa.sequence.dtw(X=feature1.T, Y=feature2.T, metric='euclidean', weights_mul=np.array([np.sqrt([2]),1,1], dtype=np.float))	#cosine rychlejsie


def Similarity(wp):
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


def SimilarityNew(wp, interval=[0.85, 1.15]):
	sim_list = []
	tmp_list = []
	good_trend = 0
	false_trend = 0
	constant_false = 0
	RIGHT = 0
	UP = 1
	DIAG = 2
	direction = -1	 # unknown at first
	prev_direction = -1
	prev_point = None
	for point in np.flip(wp,0):
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
		# print("PREVIOUS DIRECTION:", prev_direction, "	DIRECTION:", direction)
		if tmp_list != []:
			if (direction == RIGHT and prev_direction == UP) or (direction == UP and prev_direction == RIGHT):
				constant_false = 0
				# false_trend += 1
			elif (direction == RIGHT and prev_direction == RIGHT) or (direction == UP and prev_direction == UP):
				# good_trend -= 1
				false_trend += 1
				constant_false += 1
			elif (direction == DIAG):
				good_trend += 1
				constant_false = 0
			else: 
				constant_false = 0
			tmp_list.append(point)

		# print("GT:", good_trend, "FT:", false_trend, "CF:", constant_false)
		# print(tmp_list)
		if tmp_list == [] and direction == DIAG:
			if prev_direction == -1 or (prev_direction == DIAG and good_trend >= 5):
				tmp_list.append(point)
			good_trend +=1
			false_trend = 0
			constant_false = 0
		if constant_false >= 25:
			del tmp_list[-1]  	
			false_trend -= 1
			for i in range(constant_false):
				if len(tmp_list) > 0: 
					del tmp_list[-1]  
					false_trend -= 1
			if len(tmp_list) >= 150:		     
				ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
				ratio_half1 = (tmp_list[-1][0] - tmp_list[len(tmp_list)//2][0]) / (tmp_list[-1][1] - tmp_list[len(tmp_list)//2][1])
				ratio_half2 = (tmp_list[len(tmp_list)//2][0] - tmp_list[0][0]) / (tmp_list[len(tmp_list)//2][1] - tmp_list[0][1])
				ratio_third1 = (tmp_list[-1][0] - tmp_list[len(tmp_list)//3*2][0]) / (tmp_list[-1][1] - tmp_list[len(tmp_list)//3*2][1])
				ratio_third2 = (tmp_list[len(tmp_list)//3*2][0] - tmp_list[len(tmp_list)//3][0]) / (tmp_list[len(tmp_list)//3*2][1] - tmp_list[len(tmp_list)//3][1])
				ratio_third3 = (tmp_list[len(tmp_list)//3][0] - tmp_list[0][0]) / (tmp_list[len(tmp_list)//3][1] - tmp_list[0][1])
				# if interval[0] < ratio < interval[1] and \
				# 	interval[0] < ratio1 < interval[1] and \
				# 	interval[0] < ratio2 < interval[1]:
				if interval[0] < ratio < interval[1]:
					if (len(tmp_list) / false_trend > 2.3):
						print("I have constant for ft bigger than 2.3")
					print("Constant:", len(tmp_list)/false_trend )
					# if 1.0 < good_trend/false_trend:
					sim_list.append(np.array(tmp_list))
				print("ratio:", ratio, "ratio1/2:",ratio_half2, "ratio2/2:",ratio_half1)
				print("ratio:", ratio, "ratio1/3:", ratio_third3, "ratio2/3:", ratio_third2, "ratio3/3:",ratio_third1)
			constant_false = 0
			false_trend = 0
			good_trend = 0
			tmp_list = []
		prev_point = point
		prev_direction = direction

	if len(tmp_list) >= 150:
		del tmp_list[-1]  	
		false_trend -= 1
		for i in range(constant_false):
			if len(tmp_list) > 0: 
				del tmp_lis[t-1]  
				false_trend -= 1
		ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
		print(ratio)
		if interval[0] < ratio < interval[1]:
			if (len(tmp_list)/false_trend > 5):
				print("I have constant for ft bigger than 5")
			sim_list.append(np.array(tmp_list))
	sim_list = np.array(sim_list, dtype=object)
	return sim_list


def GramMatrix(feature):
	matrix = feature.dot(feature.T)#(feature.dot(feature.T)) / (np.linalg.norm(feature) * np.linalg.norm(feature.T))			
	return 0.5 * (matrix + 1)	
	# return feature


def ImageFilter(matrix, threshold=0.7, percentile=70, variance=5):
	matrix[matrix < threshold] = 0.0
	matrix[matrix >= threshold] = 1.0
	matrix = ndimage.percentile_filter(matrix, percentile=percentile, footprint=diag_matrix, mode='constant', cval=0.0)
	matrix = ndimage.gaussian_filter(matrix, variance)
	return matrix
