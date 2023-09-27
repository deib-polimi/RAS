from .application1 import Application1Normalized


class ComplexApplication(Application1Normalized):

    def __init__(self, target_cores, target_req, target_rt, sla, disturbance=0.1, init_cores=1, max_cores=5.1, weight=0.00001):
        super().__init__(sla, disturbance, init_cores)
        self.cores = target_cores
        rt = super().__computeRT__(target_req)
        self.factor = target_rt / rt
        self.A3_NOM /= self.factor
        rt = super().__computeRT__(target_req)
        self.factor = target_rt / rt
        self.cores = init_cores
        self.weight = weight
        self.max_cores=max_cores

    def __computeRT__(self, req):
        rt = super().__computeRT__(req)
        return rt * self.factor


class ApplicationModel(ComplexApplication):
    def __init__(self, sla, disturbance=0.1, init_cores=7.0, nominalRT=0):
        super().__init__(target_rt=nominalRT, target_req=100, target_cores=nominalRT*10.0, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=10.1, weight=nominalRT)
