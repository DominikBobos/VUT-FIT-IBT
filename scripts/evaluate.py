import sys

import numpy as np
import matplotlib.pyplot as plt

# import third_party_scripts.pytel.det as det
from third_party_scripts.pytel import det, probit_scale 

for system in sys.argv[1:]:
	f = open(system, "r")
	system_name = system.split('/')[-1][4:-4]
	lines = f.readlines()
	non_target = []
	target = []
	for line in lines:
		# [0] - filename 
		# [1] - class
		# [2] - hard decision
		# [3] - ratio function for DTW system
		# [-1] - soft score
		if line.split('\t')[1] == '0':	
			non_target.append(line.split('\t')[-1][:-2])
		else:
			target.append(line.split('\t')[-1][:-2])
	f.close()
	if system.find('DTW') != -1 or system.find('RTQ'):
		target = np.array(target).astype(np.float64)
		min_value = np.min(target)
		target = min_value / target
		non_target = min_value / np.array(non_target).astype(np.float64)
	else:
		target = np.array(target).astype(np.float64)
		non_target = np.array(non_target).astype(np.float64)
	d = det.Det(tar=target,non=non_target)
	d.plot_det(label=system_name)
	d.plot_min_dcf(0.5)
	# d.plot_Pmiss_at_Pfa(0.5)
	print("+===============================================+")
	print("|SYSTEM: {}\t\t\t\t|".format(system_name))
	print("|	EER: ", d.eer(), "	|")
	print("|	Pmiss_at_Pfa0.1: ", d.Pmiss_at_Pfa(0.1), "	|")
	print("|	Pmiss_at_Pfa0.3: ", d.Pmiss_at_Pfa(0.3), "	|")
	print("|	Pmiss_at_Pfa0.5: ", d.Pmiss_at_Pfa(0.5), "	|")
	print("|	Pmiss_at_Pfa0.7: ", d.Pmiss_at_Pfa(0.7), "	|")
	print("|	Pmiss_at_Pfa0.9: ", d.Pmiss_at_Pfa(0.9), "	|")
	print("+===============================================+\n")
plt.legend()
plt.show()