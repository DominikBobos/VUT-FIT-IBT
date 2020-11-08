from HTK import HTKFile
import numpy as np
import sys


#credits: https://github.com/danijel3/PyHTK

htk_reader = HTKFile()
ret = htk_reader.load(sys.argv[1])
print(htk_reader.nSamples, htk_reader.nFeatures, htk_reader.sampPeriod, htk_reader.qualifiers)
result = np.array(htk_reader.data)
print(result.shape, result)
