# read.py assumes data sits in "./spambase/rawdata/"

import numpy as np
import sys
import os

DEBUG = False

class Read:

    def __init__(self, inputs, samples, samples_test, debug=False):
        self.INPUTS = inputs
        # Split samples into training and test set
        self.SAMPLES = samples
        self.SAMPLES_TEST = samples_test
        DEBUG = debug
        # File paths
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data_raw = self.path + "/spambase/rawdata/spambase.data"
        # self.data_dat = self.path + "/spambase/data.dat"
        self.train_dat = self.path + "/spambase/train.dat"
        self.test_dat = self.path + "/spambase/test.dat"

    # Read raw data from spambase.data, shuffle, return train and test sets
    # train_set.shape: (2301, 58)
    # test_set.shape: (2300, 58)
    def read_raw(self):
        # Read data into x
        print("reading raw data...")
        x = np.loadtxt(fname=self.data_raw, delimiter=',')
        # Deep copy data and shuffle
        print("shuffling and splitting...")
        x_shuffled = x.copy()
        np.random.shuffle(x_shuffled)
        # Split data into train and test sets
        train_set = np.copy(x_shuffled[:self.SAMPLES])
        test_set = np.copy(x_shuffled[self.SAMPLES:])
        # Keep reshuffling until we get 2 balanced sets
        spam = round(self.prior(x), 4)
        while self.check_balanced(train_set, test_set, spam) == False:
            # Shuffle
            print("shuffling and splitting...")
            np.copy(np.random.shuffle(x_shuffled))
            # Split into 2 halves
            train_set = np.copy(x_shuffled[:self.SAMPLES])
            test_set = np.copy(x_shuffled[self.SAMPLES:])
        # Check shapes
        if DEBUG == True:
            print(train_set.shape)
            print(test_set.shape)
        return train_set, test_set

    # Calculate prior prob on dataset
    def prior(self, x):
        num_spam = np.sum(x[:,[self.INPUTS]])
        return num_spam / x.shape[0]

    # Make sure our split sets reflect statistics of full data set (+-~1%) 
    def check_balanced(self, x, y, spam):
        x_pos = np.sum(x[:, [57]])
        y_pos = np.sum(y[:, [57]])
        if DEBUG:
            print("x_pos",x_pos)
            print("y_pos",y_pos)
        x_pos /= self.SAMPLES
        y_pos /= self.SAMPLES_TEST
        err = 0.01
        print("  checking balance of sets falls within %s%% of original data set..." % (err*100))
        print("  bounds", spam+err, spam-err)

        if x_pos >= spam+err:
            print("    woops, x_pos was >= %s: %s" % (spam+err, x_pos))
            return False
        if x_pos <= spam-err:
            print("    woops, x_pos was <= %s: %s" % (spam-err, x_pos))
            return False
        if y_pos >= spam+err:
            print("    woops, y_pos was >= %s: %s" % (spam+err, y_pos))
            return False
        if y_pos <= spam-err:
            print("    woops, y_pos was <= %s: %s" % (spam-err, y_pos))
            return False

        print(" ",round(x_pos,4), "train set spam")
        print(" ",round(y_pos,4), "test set spam")
        print()
        return True

    def read_to_dat(self, x, y):
        # Create memmap pointer on disk and read into .dats
        # Leave room for class attribute
        fp0 = np.memmap(self.train_dat, dtype='float64',
                        mode='w+', shape=(self.SAMPLES, self.INPUTS+1))
        fp0[:] = x[:]
        del fp0
        fp1 = np.memmap(self.test_dat, dtype='float64',
                        mode='w+', shape=(self.SAMPLES_TEST, self.INPUTS+1))
        fp1[:] = y[:]
        del fp1

# def main():
#     inputs = 57
#     samples = 2301
#     samples_test = 2300

#     run = Read(inputs, samples, samples_test)
#     x, y = run.read_raw()
#     run.read_to_dat(train_set, test_set)

# if __name__ == "__main__":
#     main()
