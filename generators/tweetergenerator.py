import math
from .generator import Generator
from scipy.io import loadmat


class tweetterGen(Generator):
    
    step=None
    tweets=None
    
    def __init__(self):
        data=loadmat("/Users/emilio/git/RAS/generators/twitter_20210101_730-24_freq120sec.mat")
        self.step=data["step"]
        self.tweets=data["tweets"]

    def f(self, x):
        return self.tweets[0][x]/1000.0

    def __str__(self):
        return super().__str__()
