from .application1 import Application1
import random
from matplotlib import pyplot as plt
import numpy as np

class Application1Noisy(Application1):
    
    def __init__(self, sla, init_cores=1):
        self.t = 0
        self.cores = 1
        self.serviceTime = self.__computeRT__(1)
        super().__init__(sla, 0.0, init_cores)

    def setRT(self, req):
        exactRT = self.__computeRT__(req)

        noiseLevel = self.t // 10

        RT = max(self.serviceTime*0.5, exactRT * (1.0+np.random.normal(0, noiseLevel/10*self.serviceTime)))
        
        self.RT = RT
        self.t += 1
        return self.RT


if __name__ == "__main__":
    app=Application1Noisy(sla=0.1)
    rt=[]
    app.cores=3
    for i in range(5000):
        rt.append(app.setRT(100))
        
    plt.figure()
    plt.plot(rt)
    plt.show()

