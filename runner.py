from monitoring import Monitoring
from simulation import Simulation
from generators import Generator
from applications import Application
from controllers import StaticController
import time
import uuid

class Runner:
    def __init__(self, hrz: int, cts: list, window: int, app: Application, genMonitoring = None, name="run"):
        self.app = app
        self.sla = self.app.sla
        self.horizon = hrz
        self.controllers = cts
        self.window = window
        self.simulations = []
        self.genMonitoring = genMonitoring
        self.name = name

    def run(self, gen: Generator):
        print("*********************   %s   ********************\n" % (gen,))
        for ct in self.controllers:
            ct.setSLA(self.sla)
            if self.genMonitoring:
                m = self.genMonitoring(self.window, self.sla)
            else:
                m = Monitoring(self.window, self.sla)
            ct.setMonitoring(m)
            ct.setGenerator(gen)
            a = self.app
            
            # mi serve per far partire i controllori con un punto iniziale feasible
            if(not isinstance(ct, StaticController)):
                ct.init_cores=max(int(gen.tick(0)*0.01), 1)
                self.app.cores=max(int(gen.tick(0)*0.01), 1)
            
            s = Simulation(self.horizon, a, gen, m, ct)
            s.run()
            self.simulations.append(s)
            ct.reset()
            a.reset()
            # print()

    def log(self):
        ts = time.time()
        id = uuid.uuid1()
        f = open(f'sim-{self.name}-{ts}-{str(id)}.log', "w")
        for s in self.simulations:
            res = s.log()
            print(res)
            f.write(res)
        f.close()


    def plot(self):
        for s in self.simulations:
            s.plot()

    def getTotalViolations(self):
        print([s.getTotalViolations() for s in self.simulations])
        return sum([s.getTotalViolations() for s in self.simulations])
    
    def exportData(self):
        for s in self.simulations:
            s.exportData()
