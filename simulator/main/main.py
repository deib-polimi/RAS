from ..runner import Runner

class Main():
    def __init__(self, name, controllers, generators, horizon, monitoringWindow, app):
        self.generators = generators
        self.horizon = horizon
        self.name = name
        self.runner = Runner(horizon, controllers, monitoringWindow, app, name=name)


    def start(self):
        for generator in self.generators:
            self.runner.run(generator)
        
        self.runner.log()
        self.runner.plot()
        self.runner.exportData()
