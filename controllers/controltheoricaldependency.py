from .controller import Controller
MAX_SCALE_OUT_TIMES = 3
MIN_CORES = 0.001
MAX_CORES = 10.0

class CTControllerScaleDependency(Controller):
    def __init__(self, period, init_cores, BC=0.5, DC=0.95,max_cores=MAX_CORES, st=0.0001):
        super().__init__(period, init_cores,st)
        self.BC = BC
        self.DC = DC
        self.cores = init_cores  # new Inacio
        self.max_cores = max_cores  # new Inacio

    def reset(self):
        super().reset()
        self.xc_prec = 0

    def control(self, t):
        # getTotalRT() for Neptune and getRT for dependences
        rt = self.monitoring.getRT()
        self.rt = rt
        e = 1 / self.setpoint - 1 / rt
        self.e = e
        xc = float(self.xc_prec + self.BC * e)
        oldcores = self.cores
        self.cores = min(self.max_cores, min(max(max(MIN_CORES, oldcores / MAX_SCALE_OUT_TIMES), xc + self.DC * e),
                                        oldcores * MAX_SCALE_OUT_TIMES))
        # print('st/Cores/RT=', self.setpoint, '/', self.cores, '/', rt)
        if t < 50:
            self.cores = self.init_cores
        self.xc_prec = float(self.cores - self.BC * e)

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)
