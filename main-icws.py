from generators import *
from controllers import *
from runner import Runner
from applications import Application1

appSLA = 0.6
horizon = 1000
monitoringWindow = 10
initCores = 1

scaleXPeriod = 1
vmPeriod = 60*3
ctnPeriod = 30

c0 = StaticController(vmPeriod, 1)
c0.setName("Static (1)")
c1 = RBControllerWithCooldown(vmPeriod, initCores, step=1, cooldown=0)
c1.setName("SimpleVM")
c2 = RBControllerWithCooldown(ctnPeriod, initCores, step=1, cooldown=0)
c2.setName("SimpleCR")
c3 = RBControllerWithCooldown(vmPeriod, initCores, step=3, cooldown=0)
c3.setName("Simple (VM) - +3")
c4 = RBControllerWithCooldown(ctnPeriod, initCores, step=3, cooldown=0)
c4.setName("Simple (CTN) - +3")
c5 = StepController(vmPeriod, initCores, {
    appSLA*0.8: 0.9, appSLA*0.9: 1, appSLA: 1.1, appSLA*1.1: 1.2, appSLA*1.201: 1.3}, cooldown=0)
c5.setName("StepVM")
c6 = StepController(ctnPeriod, initCores, {
    appSLA*0.8: 0.9, appSLA*0.9: 1, appSLA: 1.1, appSLA*1.1: 1.2, appSLA*1.201: 1.3}, cooldown=0)
c6.setName("StepCR")
c7 = TargetController(vmPeriod, initCores, cooldown=0)
c7.setName("TargetVM")
c8 = TargetController(ctnPeriod, initCores, cooldown=0)
c8.setName("TargetCR")
c9 = TargetController(scaleXPeriod, initCores, cooldown=0)
c9.setName("TargetFast")
c10 = CTControllerScaleX(scaleXPeriod, initCores)
c10.setName("ScaleX")

runner = Runner(horizon, [c1, c2, c5,
                          c6, c7, c8, c9, c10], monitoringWindow, Application1(appSLA))

g = SinGen(500, 700, 200)
g.setName("SN1")
runner.run(g)

g = SinGen(1000, 1000, 100)
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