from .main import Main
from ..utils import commons as C


tuning = (7.8, 3.3)
C.SCALEX.tune(*tuning)
C.OPT.setServiceTime(C.APP_2_S_TIME)
C.JOINT.setServiceTime(C.APP_2_S_TIME)

controllers = [
    #C.SCALEX,
    C.OPT
    #C.JOINT
]

main = Main("App2_Comparison", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_2)
main.start()



