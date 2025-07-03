from .main import Main
from ..utils import commons as C

tuning = (2.0, 1.5)

C.SCALEX.tune(*tuning)
C.OPT.setServiceTime(C.APP_1_S_TIME)

controllers = [
    #C.SCALEX,
    #C.OPT,
    #C.PPO,
    #C.PPO_GUARD,
    C.IntelligentHPA
    #C.GP_INTELLIGENT_HPA
]

C.PPO.setTrain(False)
C.PPO_GUARD.setTrain(False)

for _ in range(1):
    main = Main(f"App1Noisy-IHPA", controllers, C.GEN_TRAIN_SET, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_Noisy)
    main.start()


