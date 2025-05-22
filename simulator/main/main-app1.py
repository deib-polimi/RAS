from .main import Main
from ..utils import commons as C


tuning = (2.0, 1.5)

C.SCALEX.tune(*tuning)
C.OPT.setServiceTime(C.APP_1_S_TIME)

controllers = [
   # C.SCALEX,
   # C.OPT,
    C.PPO,
]

C.PPO.setTrain(True)

for _ in range(1000):
    main = Main(f"App1-PPO", controllers, C.GEN_TRAIN_SET, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()


