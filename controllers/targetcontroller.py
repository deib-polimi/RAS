import math
from .controller import Controller


class TargetController(Controller):
    def __init__(self, period, init_cores, cooldown=60, name=None):
        super().__init__(period, init_cores, name=name)
        self.cooldown = cooldown

    def reset(self):
        super().reset()
        self.nextAction = -1

    def control(self, t):
        if t > self.nextAction:
            cores = self.cores
            r = max(-3, min(self.monitoring.getRT()/self.setpoint, 3))
            self.cores = math.ceil(self.cores*r)
            if self.cores != cores:
                self.nextAction = self.cooldown + t

    def __str__(self):
        return super().__str__() + " cooldown: %d" % (self.cooldown,)
