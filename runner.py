from application import Application
from monitoring import Monitoring
from simulation import Simulation
from generators import Generator


class Runner:
    def __init__(self, hrz: int, cts: list, window: int, sla: float):
        self.sla = sla
        self.horizon = hrz
        self.controllers = cts
        self.window = window
        self.simulations = []

    def run(self, gen: Generator):
        #print("*********************   %s   ********************\n" % (gen,))
        for ct in self.controllers:
            ct.setSLA(self.sla)
            m = Monitoring(self.window)
            ct.setMonitoring(m)
            a = Application(self.sla)
            s = Simulation(self.horizon, a, gen, m, ct)
            # print(ct)
            s.run()
            self.simulations.append(s)
            ct.reset()
            # print()

    def log(self):
        for s in self.simulations:
            print(s.log())

    def plot(self):
        for s in self.simulations:
            s.plot()
