from simulation import Simulation
from generators import *
from controllers import *
from runner import Runner
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np

stime=0.2 # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appSLA = stime*3
horizon = 1000
monitoringWindow = 1
ctPeriod = 1
appsCount = 3 
maxCores = 100

generators = [RampGen(10, 800)] * appsCount
monitorings = [Monitoring(monitoringWindow, appSLA)] * appsCount

Names=["App1","App2","App3"]
srateAvg=np.matrix([1,1,1])
initCores=np.matrix([1,1,1])
app=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=initCores,isDeterministic=False)

g = MultiGenerator(generators)
m = MultiMonitoring(monitorings)
c = CTControllerScaleXNode(1, initCores, maxCores)

simulation = Simulation(horizon, app, g, m, c)
simulation.run()
simulation.log()