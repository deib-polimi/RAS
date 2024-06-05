from .main import Main
from ..utils import commons as C


r = (0.9, 4)

C.OPT.setServiceTime(C.APP_1_S_TIME)
C.JOINT.setServiceTime(C.APP_1_S_TIME)
C.JOINT.setRange(r)

controllers = [
    C.SCALEX,
    C.OPT,
    C.JOINT
]

main = Main(f"App1_Comparison_Joint_{r[0]}_{r[1]}", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
main.start()


