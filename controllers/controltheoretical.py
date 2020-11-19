from .controller import Controller

MAX_SCALE_OUT_TIMES = 3
MIN_CORES = 1

class CTControllerScaleX(Controller):
    def __init__(self, period, init_cores, BC=0.5, DC=0.95):
        super().__init__(period, init_cores)
        self.BC = BC
        self.DC = DC

    def reset(self):
        super().reset()
        self.xc_prec = 0

    def control(self, t):
        users = self.monitoring.getUsers()
        rt = self.monitoring.getRT()
        e = 1/self.setpoint - 1/rt
        xc = float(self.xc_prec + self.BC * e)
        oldcores = self.cores
        self.cores = min(max(MIN_CORES, xc + self.DC * e), oldcores*MAX_SCALE_OUT_TIMES)
        self.xc_prec = float(self.cores - self.BC * e)

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)
