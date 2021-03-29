from generators import *
from controllers import *
from runner import Runner
from applications import ApplicationMVA, Application1
from math import ceil

stime=0.2 # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appSLA = stime*3
horizon = 1000
monitoringWindow = 1
initCores = int(ceil(1000)) #condizione iniziale che assicura un punto di partenza stabile per il sistema

scaleXPeriod = 1
OPTCTRLPeriod = 1

c1 = CTControllerScaleX(scaleXPeriod, initCores, 5, 30)
c1.setName("ScaleX")
c2 = OPTCTRL(OPTCTRLPeriod, init_cores=initCores, st=1, stime=stime)
c2.setName("OPTCTRL")
                          
runner = Runner(horizon, [c1, c2], monitoringWindow, ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))

g = SinGen(500, 700, 200)
g.setName("SN1")
runner.run(g)

g = SinGen(1000, 1001, 100)
g.setName("SN2")
runner.run(g)

g = StepGen(range(0, 1000, 100), range(0, 10000, 1000))
g.setName("SP1")
runner.run(g)

g = StepGen([50, 800, 1000], [50, 30000, 50])
g.setName("SP2")
runner.run(g)

g = RampGen(10, 800)
g.setName("RP1")
runner.run(g)

g = RampGen(20, 800)
g.setName("RP2")
runner.run(g)


runner.log()
runner.plot()
