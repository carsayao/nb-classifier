import numpy as np
import sys
import os

class train:

    def __init__(self, inputs, samples, samples_test):
        self.INPUTS = inputs
        # Split samples into training and test set
        self.SAMPLES = samples
        self.SAMPLES_TEST = samples_test
        self.train_set = np.zeros((samples, inputs+1))
        self.test_set  = np.zeros((samples_test, inputs+1))
        self.p1 = 0
        self.p0 = 0
        self.train1_mean = np.zeros(self.INPUTS+1)
        self.train0_mean = np.zeros(self.INPUTS+1)
        self.train1_std  = np.zeros(self.INPUTS+1)
        self.train0_std  = np.zeros(self.INPUTS+1)
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.train_dat = self.path + "/spambase/train.dat"
        self.test_dat  = self.path + "/spambase/test.dat"

    def load(self):
        self.train_set = np.memmap(self.train_dat, dtype="float64", mode='r',
                                   shape=(self.SAMPLES, self.INPUTS+1))
        self.test_set  = np.memmap(self.test_dat, dtype="float64", mode='r',
                                   shape=(self.SAMPLES_TEST, self.INPUTS+1))
    
    def priors(self):
        x_pos = np.sum(self.train_set[:,[self.INPUTS]])
        y_pos = np.sum(self.test_set[:,[self.INPUTS]])
        p1 = (x_pos+y_pos)/(self.SAMPLES+self.SAMPLES_TEST)
        self.p1 = p1
        p0 = 1-p1
        self.p0 = p0
        return p1, p0

    # For each feature, compute mean and std dev in the training set of the values given each class
    def mean_std(self):
        # Separate classes
        train1 = np.zeros(self.INPUTS+1)
        train0 = np.zeros(self.INPUTS+1)
        # Stack by class, find mean and std dev
        for i in range(self.SAMPLES):
            if self.train_set[i][57] == 1:
                train1 = np.vstack((train1, self.train_set[i]))
            else:
                train0 = np.vstack((train0, self.train_set[i]))
        self.train1_mean = np.mean(train1, axis=0)
        self.train0_mean = np.mean(train0, axis=0)
        self.train1_std  = np.std(train1, axis=0)
        self.train0_std  = np.std(train0, axis=0)
        # Remove 0's from std dev
        self.train1_std  = np.where(self.train1_std==0,.0001,self.train1_std)
        self.train0_std  = np.where(self.train0_std==0,.0001,self.train0_std)

def main():
    inputs = 57
    samples = 2301
    samples_test = 2300

    run = train(inputs, samples, samples_test)
    run.load()
    p1, p0 = run.priors()
    print("P(1) = %s" % p1)
    print("P(0) = %s" % p0)
    print("P(1) + P(0) = %s" % float(p1+p0))
    run.mean_std()

if __name__ == "__main__":
    main()
