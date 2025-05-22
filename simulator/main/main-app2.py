from .main import Main
from ..utils import commons as C


tuning = (15.0, 15.0)

r = (0.9, 4) # only if RL is False
RL = True
C.JOINT.setRL(RL)

if RL:
    r = (0.9, 1.1)
else:
    C.JOINT.setRange(r)


C.SCALEX.tune(*tuning)
C.JOINT.scalex.tune(*tuning)
C.JOINT.setServiceTime(C.APP_2_S_TIME)
C.OPT.setServiceTime(C.APP_2_S_TIME)

controllers = [
    C.SCALEX,
    C.OPT,
    #C.JOINT,
    C.RL
   
]

for _ in range(4):
    main = Main(f"App2_Comparison_Joint_RL_Controller", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_2)
    main.start()



