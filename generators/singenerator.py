import math
import os
if("EXTERN" in os.environ):
    from generator import Generator
else:
    from .generator import Generator


class SinGen(Generator):
    def __init__(self, mod, shift, period=100):
        super().__init__()
        self.mod = mod
        self.shift = shift
        self.period = period / (2*math.pi)

    def f(self, x):
        return abs(math.sin(x/self.period)*self.mod+self.shift)

    def __str__(self):
        return super().__str__() + " mod: %.2f shift: %.2f period %.2f" % (self.mod, self.shift, self.period*2*math.pi)
