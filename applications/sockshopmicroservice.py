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

    def __init__(self, target_cores, target_req, target_rt, sla, disturbance=0.1, init_cores=1, max_cores=2.0, weight=0.00001):
        super().__init__(sla, disturbance, init_cores)
        self.cores = target_cores
        rt = super().__computeRT__(target_req)
        self.factor = target_rt/rt
        self.A3_NOM /= self.factor
        rt = super().__computeRT__(target_req)
        self.factor = target_rt/rt
        self.cores = init_cores
        self.weight = weight
        self.max_cores = max_cores

    def __computeRT__(self, req):
        rt = super().__computeRT__(req)
        return rt * self.factor
    

class Order(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.034, target_req=100, target_cores=1.1148, sla=sla, disturbance=disturbance, init_cores=init_cores, max_cores=5.0, weight=0.0343)

class CartsDelete(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        super().__init__(target_rt=0.0667, target_req=200, target_cores=0.6313, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=1.6, weight=0.0667)
            # super().__init__(target_rt=0.0667, target_req=200, target_cores=0.6313, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.022497409739416472)


class User(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
       # super().__init__(target_rt=0.0218, target_req=300, target_cores=0.1532, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.001214907956578254)
        super().__init__(target_rt=0.0218, target_req=100, target_cores=0.1532, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=2.280, weight=0.0218)


class CartsUtil(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        # super().__init__(target_rt=0.0574, target_req=100, target_cores=0.5163, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.030692097438111413)
        super().__init__(target_rt=0.0574, target_req=200, target_cores=0.5163, sla=sla, disturbance=disturbance,
                         init_cores=init_cores, max_cores=3.42, weight=0.0574)


class Shipping(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
        # super().__init__(target_rt=0.015, target_req=100, target_cores=0.4165, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.006324267637549036)
        super().__init__(target_rt=0.015, target_req=100, target_cores=0.4165, sla=sla, disturbance=disturbance,
                         init_cores=init_cores,max_cores=2.5, weight=0.015)


class Payment(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
         # super().__init__(target_rt=0.0104, target_req=100, target_cores=0.795, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.00828921183104078)
         super().__init__(target_rt=0.0104, target_req=100, target_cores=0.795, sla=sla, disturbance=disturbance,
                          init_cores=init_cores,max_cores=2.5, weight=0.0104)


class CartsCatalogue(SockShopMicroservice):
    def __init__(self, sla, disturbance=0.1, init_cores=1):
            # super().__init__(target_rt=0.0533, target_req=100, target_cores=0.1027, sla=sla, disturbance=disturbance, init_cores=init_cores, weight=0.006171896191052099)
            super().__init__(target_rt=0.0533, target_req=200, target_cores=0.1027, sla=sla, disturbance=disturbance,
                             init_cores=init_cores, max_cores=1.5, weight=0.0533)
