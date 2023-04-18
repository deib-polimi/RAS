from generators import *
from controllers import *
from applications import Application1
from monitoring import Monitoring
from simulationlist import SimulationWithDependeciesList

stime = 0.2
appSLA = stime * 3
horizon = 2000
monitoringWindow = 10
initCores = 100
period = 1

c0 = CTControllerScaleX(period, initCores); c0.setName("ScaleX")
c1 = CTControllerScaleX(period, initCores); c1.setName("ScaleX")
c2 = CTControllerScaleX(period, initCores); c2.setName("ScaleX")
c3 = CTControllerScaleX(period, initCores); c3.setName("ScaleX")
c4 = CTControllerScaleX(period, initCores); c4.setName("ScaleX")


apps = [Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores), Application1(appSLA, init_cores=initCores)]
cts = [c0, c1, c2,c3,c4] # a controller for each App
mns = [Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA), Monitoring(monitoringWindow, appSLA)]

c0.setSLA(apps[0].sla); c1.setSLA(apps[1].sla); c2.setSLA(apps[2].sla); c3.setSLA(apps[3].sla); c4.setSLA(apps[4].sla)
c0.setMonitoring(mns[0]); c1.setMonitoring(mns[1]); c2.setMonitoring(mns[2]); c3.setMonitoring(mns[3]); c4.setMonitoring(mns[4])


g0 = SinGen(500, 700, 200); g0.setName("SN1 - App 1") # a generator for each App
g1 = RampGen(10, 800); g1.setName("RP1 - App 2")
g2 = SinGen(500, 700, 200); g2.setName("SN1 - App 2")
g3 = RampGen(10, 800); g3.setName("RP1 - App 3")
g4 = SinGen(500, 700, 200); g4.setName("SN1 - App 4")

c0.setGenerator(g0); c1.setGenerator(g1); c2.setGenerator(g2); c3.setGenerator(g3); c4.setGenerator(g4)

gens = [g0, g1, g2, g3, g4] # set to each App

simul = SimulationWithDependeciesList(horizon, apps, gens, mns, cts)
simul.run()
simul.plot()