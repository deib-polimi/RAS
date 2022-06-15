import math
from .generator import Generator
from scipy.io import loadmat


class tweetterGen(Generator):
    
    step=None
    tweets=None
    
    def __init__(self):
        data=loadmat("./generators/twitter_20210101_730-24_freq120sec.mat")
        self.step=data["step"]
        self.tweets=data["tweets"]
        self.maxIndex = len(self.tweets[0])

    def f(self, x):
        return self.tweets[0][x % self.maxIndex]/1000.0

    def __str__(self):
        return super().__str__()
