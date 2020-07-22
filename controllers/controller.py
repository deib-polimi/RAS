class Controller:
    def __init__(self, period, init_cores, st=0.8):
        self.period = period
        self.init_cores = init_cores
        self.st = st
        self.name = type(self).__name__

    def setName(self, name):
        self.name = name

    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = sla*self.st

    def setMonitoring(self, monitoring):
        self.monitoring = monitoring

    def tick(self, t):
        if not t:
            self.reset()

        if t and not (t % self.period):
            self.control(t)

        return self.cores

    def control(self, t):
        pass

    def reset(self):
        self.cores = self.init_cores

    def __str__(self):
        return "%s - period: %d init_cores: %.2f" % (self.name, self.period, self.init_cores)
