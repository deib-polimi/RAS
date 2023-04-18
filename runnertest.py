from monitoring import Monitoring
from simulation import Simulation
from generators import Generator
from applications import Application
from controllers import StaticController
import time
import uuid


class RunnerTest:
    def __init__(self, nodetest):
        self.app = nodetest.apps
        self.sla = self.app.sla
        self.horizon = nodetest.horizon
        self.controllers = nodetest.controller
        self.window = nodetest.window
        self.simulations = []
        self.genMonitoring = nodetest.genMonitoring
        self.name = nodetest.name
        self.generator = nodetest.generators
        self.nodeName=nodetest.nodeName

    def run(self):
        print("***********TEST**********   %s   ++++++++++++++±TEST++++TEST++++\n" % (self.generator,))
        for ct in self.controllers:
            ct.setSLA(self.sla)
            if self.genMonitoring:
                m = self.genMonitoring(self.window, self.sla)
            else:
                m = Monitoring(self.window, self.sla)
            ct.setMonitoring(m)
            ct.setGenerator(self.generator)
            a = self.app

            # mi serve per far partire i controllori con un punto iniziale feasible
            if (not isinstance(ct, StaticController)):
                ct.init_cores = max(int(self.generator.tick(0) * 0.01), 1)
                self.app.cores = max(int(self.generator.tick(0) * 0.01), 1)
            print('horizon=', self.horizon, ' |APP=', a, '  |generator=', self.generator, '  |m=', m, '  |ct=', ct, ' |Node=', self.nodeName)
            s = Simulation(self.horizon, a, self.generator, m, ct, self.nodeName)
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