from generators import *
from controllers import *
from runner import Runner
from applications import ApplicationMVA
import numpy


def scaleXTune():
    for BC in numpy.arange(0.1, 10, 0.5):
        for DC in numpy.arange(0.1, 10, 0.5):
            c1 = CTControllerScaleX(scaleXPeriod, initCores, BC, DC)
            c1.setName("ScaleX")
            runner = Runner(horizon, [c1], monitoringWindow,  ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))

            runAll(runner)

            v = runner.getTotalViolations()
            if v < tuning[2]:
                tuning = (BC, DC, v)
            print((BC, DC, v), tuning)
            
    return tuning[0], tuning[1]

def runAll(runner):
    # g = SinGen(25, 30, 50)
    # g.setName("SN1")
    # runner.run(g)
    #
    # g = SinGen(25, 60, 50)
    # g.setName("SN2")
    # runner.run(g)
    #
    # g = StepGen(range(0, 100, 20), range(0, 150, 30))
    # g.setName("SP1")
    # runner.run(g)
    #
    # g = StepGen([10, 50, 10], [10, 50, 10])
    # g.setName("SP2")
    # runner.run(g)
    #
    # g = RampGen(2, 80)
    # g.setName("RP1")
    # runner.run(g)
    #
    # g = RampGen(4, 60)
    # g.setName("RP2")
    # runner.run(g)
    #
    # g=tweetterGen()
    # g.setName("twetter")
    # runner.run(g)
    
    # g = SinGen(500, 700, 200)
    # g.setName("SN1")
    # runner.run(g)
    #
    # g = SinGen(1000, 1100, 100)
    # g.setName("SN2")
    # runner.run(g)
    #
    # g = StepGen(range(0, 1000, 100), range(0, 10000, 1000))
    # g.setName("SP1")
    # runner.run(g)
    #
    # g = StepGen([50, 800, 1000], [50, 5000, 50])
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
    #
    # g=tweetterGen()
    # g.setName("twetter")
    # runner.run(g)
    
    g=ibmGen()
    g.setName("ibm")
    runner.run(g)
    


stime=0.02 # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appSLA = 0.6
horizon = 1000
monitoringWindow = 10
initCores = 1 #condizione iniziale che assicura un punto di partenza stabile per il sistema

scaleXPeriod = 1
OPTCTRLPeriod = 1
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

tuning = (7.8, 3.3)

c10 = CTControllerScaleX(scaleXPeriod, initCores, tuning[0], tuning[1])
c10.setName("ScaleX")
c11 = OPTCTRL(OPTCTRLPeriod, init_cores=initCores, st=0.8, stime=stime, maxCores=10**6)
c11.setName("OPTCTRL")
                          

#runner = Runner(horizon, [c0], monitoringWindow, ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))
runner = Runner(horizon, [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], monitoringWindow, ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))
#runner = Runner(horizon, [c11], monitoringWindow, ApplicationMVA(sla=appSLA,stime=stime,init_cores=initCores))

runAll(runner)

runner.log()
#runner.plot()
runner.exportData()

