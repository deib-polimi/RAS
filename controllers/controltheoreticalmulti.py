from .controller import Controller

MAX_SCALE_OUT_TIMES = 5
MIN_CORES = 0.1

class CTControllerScaleXNode(Controller):
    def __init__(self, period, init_cores: list, max_cores, BC=0.5, DC=0.95):
        super().__init__(period, init_cores)
        #print(init_cores)
        self.BC = BC
        self.DC = DC
        self.N = len(init_cores)
        self.max_cores = max_cores

    def reset(self):
        super().reset()
        self.xc_precs = [0] * self.N

    def control(self, t):
        rts = self.monitoring.getRT()
        for i in range(self.N):
            rt = rts[i]
            e = 1/self.setpoint[i] - 1/rt
            print(i, e)
            xc = float(self.xc_precs[i] + self.BC * e)
            oldcores = self.cores[i]

            self.cores[i] = min(max(max(MIN_CORES, oldcores/MAX_SCALE_OUT_TIMES), xc + self.DC * e), oldcores*MAX_SCALE_OUT_TIMES)
            if t < 10:
                self.cores[i] = self.init_cores[i]

        allocations = sum(self.cores)
        print("desired cores", self.cores)
        if allocations > self.max_cores:
            for i in range(self.N):
                self.cores[i] = self.cores[i] * self.max_cores / allocations
        for i in range(self.N):
            self.xc_precs[i] = float(self.cores[i] - self.BC * e)
        print("actuated cores", self.cores)

    
    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = [s*self.st for s in self.sla]

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)