from ..runner import Runner
import numpy 

def scaleXTune(controller, horizon, monitoringWindow, app, generators):
    for BC in numpy.arange(0.1, 10, 0.5):
        for DC in numpy.arange(0.1, 10, 0.5):
            runner = Runner(horizon, [controller], monitoringWindow,  app)
            for g in generators:
                runner.run(g)
            v = runner.getTotalViolations()
            if v < tuning[2]:
                tuning = (BC, DC, v)
            
    return tuning[0], tuning[1]

