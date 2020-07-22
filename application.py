import random


class Application:
    A1_NOM = 0.00763
    A2_NOM = 0.0018
    A3_NOM = 0.5658

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
        return ((1000.0*self.A2_NOM+self.A1_NOM)*req+1000*self.A1_NOM*self.A3_NOM*self.cores)/(req+1000.0*self.A3_NOM*self.cores)
