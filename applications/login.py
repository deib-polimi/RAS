import math

import networkx as nx

if __name__ == "__main__":
    from application import Application
    from matplotlib import pyplot as plt
else:
    from .application import Application


class Login(Application):
    A1_NOM = 0.00073500
    A2_NOM = 0.00002967
    A3_NOM = 0.00044300
    #numberuser=0 #new

    def __computeRT__(self, req):
        return ((1000.0 * self.A2_NOM + self.A1_NOM) * req + 1000 * self.A1_NOM * self.A3_NOM * self.cores) / (
                req + 1000.0 * self.A3_NOM * self.cores)



# if __name__ == "__main__":
#     app = Application1(sla=0.1)
#     rt = []
#     app.cores = 1
#
#     for i in range(10000):
#          rt.append(i / app.__computeRT__(req=i))
#     plt.figure()
#     plt.plot(rt)
#     plt.show()