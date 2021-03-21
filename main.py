from generators import *
from controllers import *
from runner import Runner
from applications import ApplicationMVA
from math import ceil

stime=1.0 # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appSLA = stime*3
horizon = 500
monitoringWindow = 1
initCores = int(ceil(226/appSLA)) #condizione iniziale che assicura un punto di partenza stabile per il sistema

scaleXPeriod = 1
vmPeriod = 1
ctnPeriod = 30

# c0 = StaticController(vmPeriod, 1)
# c0.setName("Static (1)")
c1 = RBControllerWithCooldown(vmPeriod, initCores, step=1, cooldown=0)
c1.setName("SimpleVM-RuleBased")
c11 = OPTCTRL( vmPeriod, init_cores=initCores, st=1,stime=stime)
c11.setName("OPTCTRL")
# c2 = RBControllerWithCooldown(ctnPeriod, initCores, step=1, cooldown=0)
# c2.setName("SimpleCR")
#c3 = RBControllerWithCooldown(vmPeriod, initCores, step=3, cooldown=0)
#c3.setName("Simple (VM) - +3")
#c4 = RBControllerWithCooldown(ctnPeriod, initCores, step=3, cooldown=0)
#c4.setName("Simple (CTN) - +3")
# c5 = StepController(vmPeriod, initCores, {
    # appSLA*0.8: 0.9, appSLA*0.9: 1, appSLA: 1.1, appSLA*1.1: 1.2, appSLA*1.201: 1.3}, cooldown=0)
# c5.setName("StepVM")
# c6 = StepController(ctnPeriod, initCores, {
    # appSLA*0.8: 0.9, appSLA*0.9: 1, appSLA: 1.1, appSLA*1.1: 1.2, appSLA*1.201: 1.3}, cooldown=0)
# c6.setName("StepCR")
# c7 = TargetController(vmPeriod, initCores, cooldown=0)
# c7.setName("TargetVM")
c8 = TargetController(ctnPeriod, initCores, cooldown=0)
c8.setName("TargetCR")
# c9 = TargetController(scaleXPeriod, initCores, cooldown=0)
# c9.setName("TargetFast")
# c10 = CTControllerScaleX(scaleXPeriod, initCores)
# c10.setName("ScaleX")

# runner = Runner(horizon, [c1, c2, c5,
                          # c6, c7, c8, c9, c10], monitoringWindow, Application1(appSLA))
                          
runner = Runner(horizon, [c1,c8,c11], monitoringWindow, ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))

g = SinGen(200, 201, 50)
g.setName("SN1")
c11.generator=g; #quando questo attributo e' !=None uso la filosofia open loop per il controllo ottimo
runner.run(g)

# g = SinGen(1000, 1000, 100)
# g.setName("SN2")
# runner.run(g)
#
# g = StepGen(range(0, 10, 1), range(0, 100, 10))
# g.setName("SP1")
# runner.run(g)
#
# g = StepGen([50, 800, 1000], [50, 30000, 50])
# g.setName("SP2")
# runner.run(g)
#
# g = RampGen(10, 800)
# g.setName("RP1")
# runner.run(g)
#
# g = RampGen(20, 800)
# g.setName("RP2")
# runner.run(g)

runner.log()
runner.plot()
