import numpy as np
import sys
import os

DEBUG = False
# DEBUG = True

class Read:

    def __init__(self, inputs, samples, samples_test):
        self.INPUTS = inputs
        # Split samples into training and test set
        self.SAMPLES = samples
        self.SAMPLES_TEST = samples_test
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data_raw = self.path + "/spambase/rawdata/spambase.data"
        # self.data_dat = self.path + "/spambase/data.dat"
        self.train_dat = self.path + "/spambase/train.dat"
        self.test_dat = self.path + "/spambase/test.dat"

    # Read raw data from spambase.data, shuffle, return train and test sets
    # train_set.shape: (2301, 58)
    # test_set.shape: (2300, 58)
    def read_raw(self):
        x = np.loadtxt(fname=self.data_raw, delimiter=',')
        # Deep copy data and shuffle
        x_shuffled = x.copy()
        np.random.shuffle(x_shuffled)
        # Split data into train and test sets
        train_set = np.copy(x_shuffled[:self.SAMPLES])
        test_set = np.copy(x_shuffled[self.SAMPLES:])
        # Keep reshuffling until we get 2 balanced sets
        while self.check_balanced(train_set, test_set) == False:
            # Shuffle
            np.copy(np.random.shuffle(x_shuffled))
            # Split into 2 halves
            train_set = np.copy(x_shuffled[:self.SAMPLES])
            test_set = np.copy(x_shuffled[self.SAMPLES:])
        # Check shapes
        if DEBUG == True:
            print(train_set.shape)
            print(test_set.shape)
        return train_set, test_set

    # def shuffle(self):

    # Make sure our split sets reflect statistics of full data set (+-~1%) 
    def check_balanced(self, x, y):
        x_pos = np.sum(x[:, [57]])
        print(x_pos)
        y_pos = np.sum(y[:, [57]])
        print(y_pos)
        x_pos /= 2301
        y_pos /= 2300
        if x_pos >= .404:
            print("if x_pos >= .404:")
            print("x_pos", x_pos)
            return False
        if x_pos <= .384:
            print("if x_pos <= .384:")
            print("x_pos", x_pos)
            return False
        if y_pos >= .404:
            print("if y_pos >= .404:")
            print("y_pos", y_pos)
            return False
        if y_pos <= .384:
            print("if y_pos <= .384:")
            print("y_pos", y_pos)
            return False
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
