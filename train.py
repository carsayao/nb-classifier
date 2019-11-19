import numpy as np
from read import Read
import sys
import os

# DEBUG = False
DEBUG = True

class Classify:

    def __init__(self, inputs, samples, samples_test):
        self.INPUTS = inputs
        # Split samples into training and test set
        self.SAMPLES = samples
        self.SAMPLES_TEST = samples_test
        self.train_set = np.zeros((samples, inputs+1))
        self.test_set  = np.zeros((samples_test, inputs+1))
        self.p0 = 0
        self.p1 = 0
        self.train0_mean = np.zeros(self.INPUTS+1)
        self.train1_mean = np.zeros(self.INPUTS+1)
        self.train0_std  = np.zeros(self.INPUTS+1)
        self.train1_std  = np.zeros(self.INPUTS+1)
        self.correct = 0
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.train_dat = self.path + "/spambase/train.dat"
        self.test_dat  = self.path + "/spambase/test.dat"
    
    def load(self, x, y):
        self.train_set = np.copy(x)
        self.test_set = np.copy(y)

    def load_from_dat(self):
        self.train_set = np.memmap(self.train_dat, dtype="float64", mode='r',
                                   shape=(self.SAMPLES, self.INPUTS+1))
        self.test_set  = np.memmap(self.test_dat, dtype="float64", mode='r',
                                   shape=(self.SAMPLES_TEST, self.INPUTS+1))
    
    # Calculate probability of class
    def priors(self):
        x_pos = np.sum(self.train_set[:,[self.INPUTS]])
        y_pos = np.sum(self.test_set[:,[self.INPUTS]])
        print(x_pos+y_pos)
        p1 = (x_pos+y_pos)/(self.SAMPLES+self.SAMPLES_TEST)
        self.p1 = p1
        p0 = 1-p1
        self.p0 = p0
        return p1, p0

    # For each feature, compute mean and std dev in the training set of the values given each class
    def mean_std(self):
        # Separate classes
        train0 = np.zeros(self.INPUTS+1)
        train1 = np.zeros(self.INPUTS+1)
        # Stack by class, find mean and std dev
        for i in range(self.SAMPLES):
            if self.train_set[i][57] == 1:
                train1 = np.vstack((train1, self.train_set[i]))
            else:
                train0 = np.vstack((train0, self.train_set[i]))
        # Calculate mean and std of each class for training set
        self.train0_mean = np.mean(train0, axis=0)
        self.train0_std  = np.std(train0, axis=0)
        self.train1_mean = np.mean(train1, axis=0)
        self.train1_std  = np.std(train1, axis=0)
        # Remove 0's from std dev and replace with 0.0001
        self.train0_std  = np.where(self.train0_std==0,.0001,self.train0_std)
        self.train1_std  = np.where(self.train1_std==0,.0001,self.train1_std)
    
    # Calculate probabilities
    def prob(self, sample):
        # Retain target to later determine error
        true_class = sample[self.INPUTS]

        # Sliced up to 57 (so it doesn't include class)
        #   arr0: probability distribution that sample attribute is class 0
        #   arr1: probability distribution that sample attribute is class 1
        arr0 = np.log((1/(np.sqrt(2*np.pi)*self.train0_std[:57]))
                       *np.exp(-1*(np.square(sample[:57]-self.train0_mean[:57])
                               /(2*np.square(self.train0_std[:57])))))
        arr1 = np.log((1/(np.sqrt(2*np.pi)*self.train1_std[:57]))
                       *np.exp(-1*(np.square(sample[:57]-self.train1_mean[:57])
                               /(2*np.square(self.train1_std[:57])))))

        # Classify with priors
        #   prob0: probability sample is class 0
        #   prob1: probability sample is class 1
        prob0 = np.log(self.p0) + np.sum(arr0)
        prob1 = np.log(self.p1) + np.sum(arr1)

        # Determine class prediction and find error
        # If argmax is prob0, our prediction was 0, else prediction was 1
        if max(prob0,prob1) == prob0:
            prediction = 0
        else:
            prediction = 1
        # If error == 0, add 1 to our correctness
        if (true_class-prediction) == 0:
            self.correct += 1

        if DEBUG == True:
            print("prob0",prob0)
            print("prob1",prob1)
            print("t",true_class)
            print("y", prediction)
            print("l",true_class-prediction)
            print()

    # Nb classifier, acts like a wrapper
    def nb(self):
        # Iterate through our test set and push it through our classifier
        for i in range(self.test_set.shape[0]):
            self.prob(self.test_set[i])
        print("correct",self.correct,self.correct/2300)

        # Make sure we have the correct number of class 0 and 1
        if DEBUG == True:
            x_pos = np.sum(self.train_set[:,[self.INPUTS]])
            y_pos = np.sum(self.test_set[:,[self.INPUTS]])
            print(x_pos, y_pos)
            print("x_pos+y_pos",x_pos+y_pos)

def main():
    inputs = 57
    samples = 2301
    samples_test = 2300

    # Classifier class instance
    run = Classify(inputs, samples, samples_test)

    # Data reading class instance
    read = Read(inputs, samples, samples_test)

    # Read directly from spambase.data, shuffle,
    #   and return train and test sets
    x, y = read.read_raw()

    # Load our shuffled train and test sets into classify instance
    # run.load(x, y)

    # Read two static, preshuffled arrays as our train and test sets
    #   (for debugging)
    run.load_from_dat()
    p1, p0 = run.priors()
    print("P(1) = %s" % p1)
    print("P(0) = %s" % p0)
    print("P(1) + P(0) = %s" % float(p1+p0))
    run.mean_std()
    run.nb()

if __name__ == "__main__":
    main()
