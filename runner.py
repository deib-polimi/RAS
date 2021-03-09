from monitoring import Monitoring
from simulation import Simulation
from generators import Generator
from applications import Application

class Runner:
    def __init__(self, hrz: int, cts: list, window: int, app: Application):
        self.app = app
        self.sla = self.app.sla
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
            a = self.app
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
