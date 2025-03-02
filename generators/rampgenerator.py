from .generator import Generator


class RampGen(Generator):
    def __init__(self, slope, steady, initial=1, rampstart=0):
        super().__init__()
        self.slope = slope
        self.steady = steady
        self.initial = initial
        self.rampstart = rampstart

    def f(self, x):
        if x < self.rampstart:
            return self.initial
        if x < self.steady:
            return self.initial+self.slope*(x-self.rampstart)
        else:
            return self.initial+(self.steady-self.rampstart)*self.slope

    def __str__(self):
        return super().__str__() + " slope: %.2f steady: %d initial %d" % (self.slope, self.steady, self.initial)
