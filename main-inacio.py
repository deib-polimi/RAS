from generators import *
from controllers import *
from applications import Application1
from monitoring import Monitoring
from simulationdepgio import SimulationWithDependecies

stime = 0.2
appSLA = stime * 3
horizon = 1000
monitoringWindow = 10
initCores = 10
period = 1


c0 = CTControllerScaleX(period, initCores); c0.setName("ScaleX")
c1 = CTControllerScaleX(period, initCores); c1.setName("ScaleX")


apps = [Application1(appSLA), Application1(appSLA), Application1(appSLA)]
cts = [c0, c1]
mns = [Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA)]

c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla)
c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1])

g0 = SinGen(500, 700, 200); g0.setName("SN1 - App 1")
g1 = RampGen(10, 800); g1.setName("RP1 - App 2")
c0.setGenerator(g0); c1.setGenerator(g1)

gens = [g0, g1]

simul = SimulationWithDependecies(horizon, apps, gens, mns, cts)
simul.run()
simul.plot()