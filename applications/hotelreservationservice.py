from .application1 import Application1Normalized

class HotelReservationMicroservice(Application1Normalized):

    def __init__(self, target_cores, target_req, target_rt, sla, disturbance=0.1, init_cores=1, max_cores=2.0,
                 weight=0.00001):
        super().__init__(sla, disturbance, init_cores)
        self.cores = target_cores
        rt = super().__computeRT__(target_req)
        self.factor = target_rt / rt
        self.A3_NOM /= self.factor
        rt = super().__computeRT__(target_req)
        self.factor = target_rt / rt
        self.cores = init_cores
        self.weight = weight
        self.max_cores = max_cores

    def __computeRT__(self, req):
        rt = super().__computeRT__(req)
        return rt * self.factor


class Search(HotelReservationMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.137, target_req=100, target_cores=0.210, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=5.0, weight=0.00098)

class Profile(HotelReservationMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0475, target_req=100, target_cores=0.140, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=2.35, weight=0.00138)

class Geo(HotelReservationMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0395, target_req=100, target_cores=0.140, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=1.0, weight=0.00136)

class Rate(HotelReservationMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.034, target_req=100, target_cores=0.190, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=1.0, weight=0.00134)
