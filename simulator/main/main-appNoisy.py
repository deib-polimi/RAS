from .main import Main
from ..utils import commons as C


tuning = (10.0, 10.0)

r = (0.9, 4) # only if RL is False
RL = False
C.JOINT.setRL(RL)

if RL:
    r = (0.9, 1.1)
else:
    C.JOINT.setRange(r)


C.SCALEX.tune(*tuning)
C.JOINT.scalex.tune(*tuning)
C.JOINT.setServiceTime(C.APP_2_S_TIME)
C.OPT.setServiceTime(C.APP_2_S_TIME)
C.ROBUST.setServiceTime(C.APP_2_S_TIME)

controllers = [
    C.SCALEX,
    C.OPT,
    C.ROBUST,
    #C.JOINT,
    #C.RL 
]+C.CONTROLLER_SET_INDUSTRY

for _ in range(1):
    main = Main(f"AppNoisy_SEAMS2rnd", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_Noisy)
    main.start()



