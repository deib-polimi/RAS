from .main import Main
from ..utils import commons as C


tuning = (2.0, 1.5)

C.SCALEX.tune(*tuning)

controllers = [
    C.SCALEX,
]

for _ in range(1):
    main = Main(f"SimpleApp-CT", controllers, C.GEN_SET_1, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()



