from simulation import Simulation
from generators import *
from controllers import *
from runner import Runner
from applications import AppsCluster
from math import ceil
from monitoring import Monitoring, MultiMonitoring
import numpy as np
from commons import SN1, SN2, SP2, RP1, RP2, ALL
from itertools import combinations
import sys

name = sys.argv[0].split('.')[0]

stimes=[0.1, 0.4] # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appsCount = len(stimes)
appsSLA = [x*2 for x in stimes]
horizon = 300
monitoringWindow = 1
ctPeriod = 1
maxCores = 200000

Names=[f'App{i}' for i in range(1, appsCount+1)]
srateAvg=[1.0/stime for stime in stimes]
initCores=[1 for _ in stimes]
app=AppsCluster(appNames=Names,srateAvg=srateAvg,initCores=initCores,isDeterministic=False)
AppsCluster.sla=appsSLA


c1 = CTControllerScaleXNode(1, initCores, maxCores, BCs=3, DCs=15)
c2 = OPTCTRL(monitoringWindow, init_cores=initCores, st=0.8, stime=[1/stimes[i] for i in range(appsCount)],maxCores=maxCores)
c2.setName("OPTCTRL")
c2.reset()



runner = Runner(horizon, [c2], monitoringWindow, app, lambda window, sla: MultiMonitoring([Monitoring(monitoringWindow, appsSLA[i]) for i in range(appsCount)]), name=name)
g = MultiGenerator([RP1, RP1])
runner.run(g)

# g = MultiGenerator([SN1, SN1])
# runner.run(g)
#
# g = MultiGenerator([SP1, SP1])
# runner.run(g)

runner.log()
runner.plot()