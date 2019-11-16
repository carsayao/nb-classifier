import numpy as np
import sys
import os

class read:

    def __init__(self, inputs, samples, samples_test):
        self.INPUTS = inputs
        # Split samples into training and test set
        self.SAMPLES = samples
        self.SAMPLES_TEST = samples_test
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.data_raw = self.path + "/spambase/rawdata/spambase.data"
        self.data_dat = self.path + "/spambase/data.dat"

    def read_raw(self):
        x = np.loadtxt(fname=self.data_raw, delimiter=',')
        train_set = np.copy(x[:self.SAMPLES])
        test_set = np.copy(x[self.SAMPLES:])
        x_shuffled = np.copy(np.random.shuffle(x))
        while self.check_balanced(train_set, test_set) == False:
            x_shuffled = np.copy(np.random.shuffle(x))
            train_set = np.copy(x[:self.SAMPLES])
            test_set = np.copy(x[self.SAMPLES:])
        # x_pos = np.sum(x[:, [57]])
        # print(x_pos/4601)
        print("x: ", x.shape)
        # Split into 2 halves
        #   train_set.shape: (2301, 58)
        #   test_set.shape: (2300, 58)

    # Make sure our split sets reflect statistics of full data set (+-~1%) 
    def check_balanced(self, x, y):
        x_pos = np.sum(x[:, [57]]) / 2301
        print(x_pos)
        y_pos = np.sum(y[:, [57]]) / 2300
        print(y_pos)
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

    # def read_to_dat(self):
        

def main():
    inputs = 57
    samples = 2301
    samples_test = 2300

    run = read(inputs, samples, samples_test)
    run.read_raw()

if __name__ == "__main__":
    main()
