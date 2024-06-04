from .controller import Controller


class CTControllerScaleX(Controller):
    MAX_SCALE_OUT_TIMES = 3

    def __init__(self, period, init_cores, BC=0.5, DC=0.95,st=0.8, min_cores=1, max_cores=10**8, name=None):
        super().__init__(period, init_cores,st=st, name=name)
        self.BC = BC
        self.DC = DC
        self.min_cores = min_cores
        self.max_cores = max_cores


    def reset(self):
        super().reset()
        self.xc_prec = 0
    
    def tune(self, BC, DC):
        self.BC = BC
        self.DC = DC

    def control(self, t):
        rt = self.monitoring.getRT()
        e = 1/self.setpoint - 1/rt
        xc = float(self.xc_prec + self.BC * e)
        oldcores = self.cores
        self.cores = min(max(max(self.min_cores, oldcores/self.MAX_SCALE_OUT_TIMES), xc + self.DC * e), oldcores*self.MAX_SCALE_OUT_TIMES)
        if t < 10:
            self.cores = self.init_cores
        self.xc_prec = float(self.cores - self.BC * e)
    

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)



class CTControllerScaleXJoint(CTControllerScaleX):
    
    def control(self, _):
        rt = self.monitoring.getRT()
        e = 1/self.setpoint - 1/rt
        xc = float(self.xc_prec + self.BC * e)
        self.cores = min(max(self.min_cores, xc + self.DC * e), self.max_cores)
        self.xc_prec = float(self.cores - self.BC * e)

