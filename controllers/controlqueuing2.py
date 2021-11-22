from .controller import Controller
import numpy as np


class OPTCTRL2(Controller):
    
    
    def __init__(self, period, init_cores,maxCores,st=1.0):
        super().__init__(period, init_cores,st)
        self.maxCores=maxCores
        
    def control(self, t):
        
        if(self.generator!=None):
            users=self.generator.f(t+1)    
        else:
            users=int(self.monitoring.getUsers())
            
        #risolvo il problema di controllo ottimo
        self.cores=np.minimum((1.0*users)/self.st,self.maxCores)
        print(self.cores,t,users,self.monitoring.getRT())
    
    def reset(self):
        super().reset()
    
    def setSLA(self, sla):
        self.sla = sla
        self.setpoint = self.sla*self.st
      

    def __str__(self):
        return super().__str__() + " OPTCTRL: %.2f, l: %.2f h: %.2f " % (self.step, self.l, self.h)
