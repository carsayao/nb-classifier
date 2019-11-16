import numpy as np
import os
import sys

INPUTS = 57
SAMPLES = 2301
SAMPLES_TEST = 2300
path = os.path.dirname(os.path.realpath(__file__))
data_raw = path + "/spambase/rawdata/spambase.data"
data_dat = path + "/spambase/data.dat"

arg = sys.argv[1]

if arg == 'sum_array':
    a = np.array([[1,2,3,0],
                  [1,2,3,1],
                  [1,2,3,0]])
    col_to_sum = a[:, [3]]
    print(np.sum(col_to_sum))
    # np.sum(a, axis=1)