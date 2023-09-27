from .application import Application


class Application1(Application):
    A1_NOM = 0.00763 # amplitude
    A2_NOM = 0.0018
    A3_NOM = 0.5658 # slope

    def __computeRT__(self, req):
        return ((1000.0 * self.A2_NOM + self.A1_NOM) * req + 1000 * self.A1_NOM * self.A3_NOM * self.cores) / (
                req + 1000.0 * self.A3_NOM * self.cores)



class Application1Normalized(Application1):
    def __init__(self, sla, disturbance=0.1, init_cores=1, users=0):
        super().__init__(sla, disturbance, init_cores, users)
        self.max = super().__computeRT__(100000)
        self.min = super().__computeRT__(0)

    def norm(self, x):
        return (x - self.min)/(self.max - self.min)

    
    def __computeRT__(self, req):
       rt = super().__computeRT__(req)
       return self.norm(rt)

