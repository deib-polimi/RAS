from .controller import Controller


class RBController(Controller):
    def __init__(self, period, init_cores, step=1, l=0.5, h=0.9):
        super().__init__(period, init_cores)
        self.l = l
        self.h = h
        self.step = step

    def control(self, t):
        rt = self.monitoring.getRT()
        if rt < self.l*self.sla:
            self.cores = max(1, self.cores-self.step)
        elif rt > self.h*self.sla:
            self.cores += self.step

    def __str__(self):
        return super().__str__() + " step: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)


class RBControllerWithCooldown(RBController):
    def __init__(self, period, init_cores, step=1, l=0.5, h=0.9, cooldown=60):
        super().__init__(period, init_cores, step, l, h)
        self.cooldown = cooldown

    def reset(self):
        super().reset()
        self.nextAction = -1

    def control(self, t):
        if t > self.nextAction:
            cores = self.cores
            super().control(t)
            if cores != self.cores:
                self.nextAction = t + self.cooldown

    def __str__(self):
        return super().__str__() + " cooldown: %d" % (self.cooldown,)
