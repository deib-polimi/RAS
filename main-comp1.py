from generators import *
from controllers import *
from runner import Runner
from applications import Application1

appSLA = 0.0432
horizon = 1000
monitoringWindow = 1
initCores = 1000 #condizione iniziale che assicura un punto di partenza stabile per il sistema

scaleXPeriod = 1
OPTCTRLPeriod = 1

c1 = CTControllerScaleX(scaleXPeriod, initCores, 5, 30)
c1.setName("ScaleX")
c2 = OPTCTRL2(OPTCTRLPeriod, init_cores=initCores, st=4,maxCores=10000000)
c2.setName("OPTCTRL")
                          
runner = Runner(horizon, [c1], monitoringWindow, Application1(appSLA))

# g = SinGen(500, 700, 200)
# g.setName("SN1")
# runner.run(g)

# g = SinGen(1000, 1001, 100)
# g.setName("SN2")
# runner.run(g)
#
# g = StepGen(range(0, 1000, 100), range(0, 10000, 1000))
# g.setName("SP1")
# runner.run(g)
#
# g = StepGen([50, 800, 1000], [50, 30000, 50])
# g.setName("SP2")
# runner.run(g)
#
g = RampGen(10, 800)
g.setName("RP1")
runner.run(g)
#
# g = RampGen(20, 800)
# g.setName("RP2")
# runner.run(g)


runner.log()
runner.plot()
