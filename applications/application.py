import random

class Application:
    def __init__(self, sla, disturbance=0.1):
        self.cores = 1
        self.RT = 0.0
        self.sla = sla
        self.disturbance = disturbance

    def setRT(self, req):
        exactRT = self.__computeRT__(req)
        self.RT = exactRT * (1.0+random.random()*self.disturbance)
        return self.RT

    def __computeRT__(self, req):
        pass

