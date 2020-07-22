from .generator import Generator


class RampGen(Generator):
    def __init__(self, slope, steady, initial=0):
        self.slope = slope
        self.steady = steady
        self.initial = initial

    def f(self, x):
        if x < self.steady:
            return self.slope*x
        else:
            return self.steady*self.slope

    def __str__(self):
        return super().__str__() + " slope: %.2f steady: %d initial %d" % (self.slope, self.steady, self.initial)
