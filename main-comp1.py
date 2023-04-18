from generators import *
from controllers import *
from runner import Runner
from applications import Application1
import numpy 
import os
runnerlist=[]

def scaleXTune(): # Has some issues on variables
    for BC in numpy.arange(0.1, 10, 0.5):
        for DC in numpy.arange(0.1, 10, 0.5):
            c1 = CTControllerScaleX(scaleXPeriod, initCores, BC, DC)
            c1.setName("ScaleX")
            runner = Runner(horizon, [c1], monitoringWindow,  Application1(sla=appSLA, stime=stime, init_cores=initCores))
            runAll(runner)

            v = runner.getTotalViolations()
            if v < tuning[2]:
                tuning = (BC, DC, v)
            print((BC, DC, v), tuning)
            
    return tuning[0], tuning[1]


def runAll(runner):
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
    
stime=0.2 # average service time of the MVA application (this is required by both the MVA application and the OPTCTRL)
appSLA = stime*3
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

tuning = (8, 1)

setpoints=[0.8]


for st in setpoints:

    c10 = CTControllerScaleX(scaleXPeriod, initCores, tuning[0], tuning[1],st=st)
    c10.setName("ScaleX")
    c11 = OPTCTRL(OPTCTRLPeriod, init_cores=initCores, st=st, stime=stime, maxCores=10**6)
    c11.setName("OPTCTRL")
                              
    
    #runner = Runner(horizon, [c0], monitoringWindow, Application1(appSLA))
    runner = Runner(horizon, [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10], monitoringWindow, Application1(appSLA))
    #runner = Runner(horizon, [c10], monitoringWindow, Application1(appSLA))
    #runner = Runner(horizon, [c11], monitoringWindow, Application1(appSLA))
    
    runAll(runner)
    
    runner.log()
    runner.plot()
    runner.exportData()
    
    #os.rename('./experiments/matfile/OPTCTRL-SN1.mat', './experiments/matfile/OPTCTRL-SN1-%.2f.mat'%(st))
    

