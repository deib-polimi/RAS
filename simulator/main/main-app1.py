from .main import Main
from ..utils import commons as C


r = (0.9, 4) # only if RL is False
RL = False
C.JOINT.setRL(RL)

if RL:
    r = (0.9, 1.1)
else:
    C.JOINT.setRange(r)


C.OPT.setServiceTime(C.APP_1_S_TIME)
C.JOINT.setServiceTime(C.APP_1_S_TIME)


controllers = [
    #C.SCALEX,
    #C.OPT,
    #C.JOINT,
    C.RL
]

for _ in range(100):
    main = Main(f"App1_Comparison_RL_Controller", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()


