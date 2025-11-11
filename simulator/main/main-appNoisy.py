from .main import Main
from ..utils import commons as C



controllers = [
    C.PPO,
    C.PPO_FUZZY
]

C.PPO.setTrain(False)
C.PPO_FUZZY.setTrain(False)

for _ in range(1):
    main = Main(f"App1Noisy-PPO", controllers, C.GEN_TRAIN_SET, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()


