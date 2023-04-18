from runnertest import RunnerTest


class RunnerWithDependences:

    def __init__(self, nodeList):
        self.nodelist = nodeList
        self.runList = []
        for node in self.nodelist:
            runner =RunnerTest(node)
            self.runList.append(runner)

    def runDep(self):
        for runner in self.runList:
            print('RUNLIST VALUES ', runner.name)
            runner.run()

    def logDep(self):
        for runner in self.runList:
            runner.log()

    def plotDep(self):
        for runner in self.runList:
            runner.plot()

    def getTotalViolationsDep(self):
        violations = []
        for runner in self.runList:
            violations.append(runner.getTotalViolations())
        return violations

    def exportDataDep(self):
        for runner in self.runList:
            runner.getTotalViolations()