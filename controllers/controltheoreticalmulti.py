from .controller import Controller

MAX_SCALE_OUT_TIMES = 10000000
MIN_CORES = 0.1

class CTControllerScaleXNode(Controller):
    def __init__(self, period, init_cores: list, max_cores, BCs, DCs):
        super().__init__(period, init_cores)
        #print(init_cores)
        self.BCs = BCs
        self.DCs = DCs
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
            print(f'app {i} error:', e)
            #xc = float(self.xc_precs[i] + self.BC * e)
            oldcores = self.cores[i]

            #self.cores[i] = min(max(max(MIN_CORES, oldcores/MAX_SCALE_OUT_TIMES), xc + self.DC * e), oldcores*MAX_SCALE_OUT_TIMES)
            self.cores[i] = max(0.001, self.DCs[i]*e)
          

        allocations = sum(self.cores)
        print("desired cores", self.cores)
        if allocations > self.max_cores:
            for i in range(self.N):
                self.cores[i] = self.cores[i] * self.max_cores / allocations
        #for i in range(self.N):
        #    self.xc_precs[i] = float(self.cores[i] - self.BC * e)
        print("actuated cores", self.cores)

    
    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = [s*self.st for s in self.sla]

    def __str__(self):
        return super().__str__() + " BC: %.2f DC: %.2f " % (self.BC, self.DC)