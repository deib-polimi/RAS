from .application1 import Application1Normalized


# class SockShopMicroserviceOld(Application):

#     def __init__(self, sla, disturbance=0.1, init_cores=1, a1=0.00763, a2=0.0018, a3=0.0018):
#         super().__init__(sla, disturbance, init_cores)
#         self.A1_NOM = a1
#         self.A2_NOM = a2
#         self.A3_NOM = a3

#     def __computeRT__(self, req):
#         rt = ((1000.0 * self.A2_NOM + self.A1_NOM) * req + 1000 * self.A1_NOM * self.A3_NOM * self.cores) / (
#                 req + 1000.0 * self.A3_NOM * self.cores)
#         return rt


class SockShopMicroservice(Application1Normalized):

    def __init__(self, target_cores, target_req, target_rt, sla, disturbance=0.1, init_cores=1):
        super().__init__(sla, disturbance, init_cores)
        self.cores = target_cores
        rt = super().__computeRT__(target_req)
        self.factor = target_rt/rt
        self.A3_NOM /= self.factor
        rt = super().__computeRT__(target_req)
        self.factor = target_rt/rt
        self.cores = init_cores 

    def __computeRT__(self, req):
        rt = super().__computeRT__(req)
        return rt * self.factor
    

class Order(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0343, target_req=100, target_cores=1.1148, sla=sla, disturbance=disturbance, init_cores=init_cores)


class CartsDelete(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0613, target_req=200, target_cores=0.6313, sla=sla, disturbance=disturbance, init_cores=init_cores)


class User(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.03408, target_req=300, target_cores=0.1532, sla=sla, disturbance=disturbance, init_cores=init_cores)


class CartsUtil(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0574, target_req=100, target_cores=0.5163, sla=sla, disturbance=disturbance, init_cores=init_cores)


class Shipping(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.015, target_req=100, target_cores=0.4165, sla=sla, disturbance=disturbance, init_cores=init_cores)


class Payment(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0104, target_req=100, target_cores=0.795, sla=sla, disturbance=disturbance, init_cores=init_cores)


class CartsCatalogue(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0533, target_req=100, target_cores=0.1027, sla=sla, disturbance=disturbance, init_cores=init_cores)