from .main import Main
from ..utils import commons as C


tuning = (7.8, 3.3)
r = (0.8, 1.2)

C.SCALEX.tune(*tuning)
C.OPT.setServiceTime(C.APP_2_S_TIME)
C.JOINT.setServiceTime(C.APP_2_S_TIME)
C.JOINT.scalex.tune(*tuning)
C.JOINT.setRange(r)

controllers = [
    C.SCALEX,
    C.OPT,
    C.JOINT
]

main = Main(f"App2_Comparison_Joint_{r[0]}_{r[1]}", controllers, [C.GEN_SET_1[0]], C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_2)
main.start()



