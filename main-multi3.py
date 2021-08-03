from simulation import Simulation
from generators import *
from controllers import *
from runner import Runner
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np
from commons import SN1, SN2, SP1, SP2, RP1, RP2, ALL
from itertools import combinations
import sys

name = sys.argv[0].split('.')[0]

stimes=[0.1, 0.4] # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appsCount = len(stimes)
appsSLA = [x*2 for x in stimes]
horizon = 200
monitoringWindow = 1
ctPeriod = 1
maxCores = 200000

Names=[f'App{i}' for i in range(1, appsCount+1)]
srateAvg=[1.0/stime for stime in stimes]
initCores=[10 for _ in stimes]
app=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=initCores,isDeterministic=False)
AppsCluster.sla=appsSLA


c1 = CTControllerScaleXNode(1, initCores, maxCores, BC=3, DC=15)
c2 = OPTCTRL(monitoringWindow, init_cores=initCores, st=0.8, stime=[1/stimes[i] for i in range(appsCount)],maxCores=maxCores)
c2.setName("OPTCTRL")
c2.reset()



runner = Runner(horizon, [c2, c1], monitoringWindow, app, lambda window, sla: MultiMonitoring([Monitoring(monitoringWindow, appsSLA[i]) for i in range(appsCount)]), name=name)
g = MultiGenerator([SN2, SN2])
#runner.run(g)

g = MultiGenerator([RP2, RP2])
#runner.run(g)

g = MultiGenerator([SP2, SP2])
runner.run(g)

runner.log()
runner.plot()