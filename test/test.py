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

if arg=='std':
    a = np.array([[1,2,3,0],
                  [1,2,3,1],
                  [1,2,3,0]])
    print(a)
    a = np.vstack((a, [1,2,3,4]))
    print(a)

if arg=='name_array':
    a = np.array(["trainmean", [[1,2,3,4],[5,6,7,8]]],
                  dtype=[('name', 'U10'), ('number', 'int8')])
    print(a)

if arg=='prob':
        # print((1/(np.sqrt(2*np.pi)*self.train0_std))*np.exp(-1*(np.square(sample-self.train0_mean)/np.square(2*self.train0_std))))
        print((1/(np.sqrt(2*np.pi)*1.8))*np.exp(-1*(np.square(5.2-4.8)/np.square(2*1.8))))