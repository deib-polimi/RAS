import math
from .generator import Generator
from scipy.io import loadmat


class ibmGen(Generator):
    
    step=None
    tweets=None
    
    def __init__(self):
        data=loadmat("./generators/ibm_20170902_00-24_freq180sec.mat")
        self.step=data["step"]
        self.tweets=data["num_customers"]
        self.maxIndex = len(self.tweets[0])

    def f(self, x):
        return self.tweets[0][x % self.maxIndex]

    def __str__(self):
        return super().__str__()
