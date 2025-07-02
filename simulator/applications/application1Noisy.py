from .application1 import Application1
import random
from matplotlib import pyplot as plt
import numpy as np

class Application1Noisy(Application1):
    
    def __init__(self, sla, init_cores=1):
        self.cores = 1
        self.serviceTime = self.__computeRT__(1, 0)
        super().__init__(sla, 0.0, init_cores)

    def setRT(self, req, t):
        self.RT = self.__computeRT__(req, t)

        if False:
            return self.RT * 20
        # if t >= 800 and t < 1300:
        #     return self.RT * 15
        # if t >= 1300:
        #    return self.RT * 20
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

