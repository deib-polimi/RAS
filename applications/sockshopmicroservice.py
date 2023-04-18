from .application import Application


class SockShopMicroservice(Application):

    def __init__(self, sla, disturbance=0.1, init_cores=1, a1=0.00763, a2=0.0018, a3=0.0018):
        super().__init__(sla, disturbance, init_cores)
        self.A1_NOM = a1
        self.A2_NOM = a2
        self.A3_NOM = a3

    def __computeRT__(self, req):
        rt = ((1000.0 * self.A2_NOM + self.A1_NOM) * req + 1000 * self.A1_NOM * self.A3_NOM * self.cores) / (
                req + 1000.0 * self.A3_NOM * self.cores)
        return rt
