from .main import Main
from ..utils import commons as C


tuning = (8, 1)
C.SCALEX.tune(*tuning)

C.OPT.setServiceTime(C.APP_1_S_TIME)

controllers = [
    #C.SCALEX,
    C.OPT
]

main = Main("App1_Comparison", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
main.start()


