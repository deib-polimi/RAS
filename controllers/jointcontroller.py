
from controllers import OPTCTRL, CTControllerScaleXJoint
import numpy as np


class JointController(OPTCTRL):
    def __init__(self, period, init_cores, range=(0.9, 1.6), maxCores=1000, st=0.8, name=None):
        super().__init__(period, init_cores, maxCores, st, name=name)
        self.scalex = CTControllerScaleXJoint(period, init_cores, st=st, max_cores=maxCores)
        self.cont = 0
        self.qn_cores = 0
        self.range = range

    def control(self, t):
        #if self.cont % 1 == 0:
        if True:
            self.cont = 0
            super().control(t)
            self.qn_cores = self.cores
        
        rt = self.monitoring.getRT()

        self.scalex.min_cores = self.qn_cores*self.range[0]
        self.scalex.max_cores = self.qn_cores*self.range[1]
        self.cores = self.scalex.tick(t)
        self.cont += 1

    def setMonitoring(self,monitoring):
        super().setMonitoring(monitoring)
        self.scalex.setMonitoring(self.monitoring)

    def setSLA(self,sla):
        super().setSLA(sla)
        self.scalex.setSLA(sla)

    def reset(self):
        super().reset()
        self.scalex.reset()

    def setRange(self, range):
        self.range = range

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)