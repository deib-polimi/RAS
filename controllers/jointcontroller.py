from controllers import OPTCTRL, CTControllerScaleX


class JointController(OPTCTRL):
    def __init__(self, period, init_cores, stime, BC=0.5, DC=0.95, maxCores=1000, st=0.8):
        super().__init__(period, init_cores, stime, maxCores, st)
        self.scalex = CTControllerScaleX(period, init_cores, BC, DC, st, max_cores=maxCores)
        self.cont = 0
        self.qn_cores = 0

    def control(self, t):
        #if self.cont % 1 == 0:
        if(True):
            self.cont = 0
            super().control(t)
            self.qn_cores = self.cores

        self.scalex.min_cores = self.qn_cores*0.8
        self.scalex.max_cores = self.qn_cores*1.2
        self.scalex.control(t)
        self.cores = self.scalex.cores
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

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)