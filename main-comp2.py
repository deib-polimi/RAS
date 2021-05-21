from generators import *
from controllers import *
from runner import Runner
from applications import Application1
from estimator import QNEstimaator
import  numpy as np


appSLA = 0.6
stime=appSLA/3; # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)

horizon = 1000
monitoringWindow = 10
initCores = 3

scaleXPeriod = 1
OPTCTRLPeriod = 1


c1 = CTControllerScaleX(scaleXPeriod, initCores)
c1.setName("ScaleX")
c2 = OPTCTRL(OPTCTRLPeriod, init_cores=initCores, st=1.0, stime=stime)
c2.setName("OPTCTRL")
runner = Runner(horizon, [c1, c2], monitoringWindow, Application1(appSLA))


g = SinGen(500, 700, 200)
c2.stime=appSLA/3
c2.resetEstimate()
c2.generator=g
g.setName("SN1")
runner.run(g)
#
# g = SinGen(1000, 1000, 100)
# c2.stime=appSLA/3
# c2.resetEstimate()
# c2.generator=g
# g.setName("SN2")
# runner.run(g)
#
# g = StepGen(range(0, 1000, 100), range(0, 10000, 1000))
# c2.stime=appSLA/3
# c2.resetEstimate()
# c2.generator=g
# g.setName("SP1")
# runner.run(g)
#
# g = StepGen([50, 800, 1000], [50, 30000, 50])
# c2.stime=appSLA/3
# c2.resetEstimate()
# c2.generator=g
# g.setName("SP2")
# runner.run(g)
#
# g = RampGen(10, 800)
# c2.stime=appSLA/3
# c2.resetEstimate()
# c2.generator=g
# g.setName("RP1")
# runner.run(g)
#
# g = RampGen(20, 800)
# c2.stime=appSLA/3
# c2.resetEstimate()
# c2.generator=g
# g.setName("RP2")
# runner.run(g)

runner.log()
runner.plot()