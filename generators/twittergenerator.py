from .tracegenerator import TraceGen
from scipy.io import loadmat


class TweetGen(TraceGen):
    
    def __init__(self, shift=436.5, bias = 385):
        data=loadmat("./generators/twitter_20210101_730-24_freq120sec.mat")["tweets"][0]
        super().__init__(data, shift, bias)

    def __str__(self):
        return super().__str__()
