import random
import numpy as np

class Application:
    
    def __init__(self, sla, disturbance=0.0, init_cores=1):
        self.init_cores = init_cores
        self.cores = init_cores
        self.RT = 0.0
        self.sla = sla
        self.disturbance = disturbance
        

    def setRT(self, req, t):
        exactRT = self.__computeRT__(req, t)
        self.RT = exactRT * (1.0+random.random()*self.disturbance)
        return self.RT

    def __computeRT__(self, req, t):
        pass
    
    def reset(self):
        self.cores = self.init_cores
        for i in range(0, 5):
            self.setRT(1, 0)

