from .main import Main
from ..utils import commons as C


tuning = (2.0, 1.5)

C.SCALEX.tune(*tuning)
C.OPT.setServiceTime(C.APP_1_S_TIME)

controllers = [
    C.SCALEX,
    C.OPT
] + C.CONTROLLER_SET_INDUSTRY


'''
for c in controllers[-1:]:
    c.setTrain(True)

for _ in range(1000):
    main = Main(f"PPO-train", controllers[-1:], C.GEN_TRAIN_SET, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()
'''




main = Main(f"RTT", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
main.start()


