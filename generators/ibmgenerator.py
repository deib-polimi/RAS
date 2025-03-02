from .tracegenerator import TraceGen
from scipy.io import loadmat


class IBMGen(TraceGen):
    
    def __init__(self, shift=0, bias = 1):
        data=loadmat("./generators/ibm_20170902_00-24_freq180sec.mat")["num_customers"][0]
        super().__init__(data, shift, bias)

    def __str__(self):
        return super().__str__()
