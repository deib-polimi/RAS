from simulation import Simulation
from generators import *
from controllers import *
from runner import Runner
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np

stimes=[0.1,0.4] # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appsSLA = [stimes[0],stimes[1]*2]
horizon = 200
monitoringWindow = 1
ctPeriod = 1
appsCount = 2 
maxCores = 2000


generators = [RampGen(10, 800)] * appsCount
monitorings = [Monitoring(monitoringWindow, appsSLA[i]) for i in range(appsCount)] 

Names=["App1","App2"]
srateAvg=[1.0/stime for stime in stimes]
initCores=[1,1]
app=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=initCores,isDeterministic=False)
AppsCluster.sla=appsSLA

g = MultiGenerator(generators)
m = MultiMonitoring(monitorings)

# c = CTControllerScaleXNode(1, initCores, maxCores)
# c.setSLA(appsSLA)
# c.setMonitoring(m)
# c.setGenerator(g)

c2 = OPTCTRL(monitoringWindow, init_cores=initCores, st=1, stime=[1/stimes[i] for i in range(appsCount)],maxCores=maxCores)
c2.setName("OPTCTRL")

c2.resetEstimate()
c2.setSLA(appsSLA)
c2.setGenerator(g)
c2.setMonitoring(m)

simulation = Simulation(horizon, app, g, m, c2)
simulation.run()
print(simulation.log())
simulation.plot()
