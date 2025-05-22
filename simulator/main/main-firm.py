from .main import Main
from ..utils import commons as C

from controllers.firmcontroller import FIRM_DDPG_Controller


ctrl = FIRM_DDPG_Controller(
    period=1.0,
    init_cores=4,
    min_cores=2,
    max_cores=32,
    st=0.8,
    train=True,               # False = inference
    log_dir="./logs"
)


controllers = [
    ctrl
]


for _ in range(10000):
    main = Main(f"App1-Firm", controllers, C.GEN_TRAIN_SET, C.HORIZON, C.MONITORING_WINDOW, C.APPLICATION_1)
    main.start()



