from generators import *
from controllers import *
from ..applications import *

# CONSTANTS

HORIZON = 1000
MONITORING_WINDOW = 1
INIT_CORES = 5
MIN_CORES = 1
MAX_CORES = 128
SCALEX_PERIOD = 1
OPTCTRL_PERIOD = 1
VM_PERIOD = 60*3
CONTAINER_PERIOD = 30
SET_POINT_FACTOR = 0.8
APP_1_S_TIME=0.2 
APP_2_S_TIME=0.2
APP_MMC_S_TIME=0.2 
APP_SLA = 0.4

# GENERATORS
GEN_SET_1 = [
    SinGen(500, 700, 200), 
    SinGen(1000, 1100, 100),
    StepGen(range(0, 1000, 100), range(0, 10000, 1000)),
    StepGen([50, 800, 1000], [50, 5000, 50]),
    RampGen(10, 800),
    RampGen(20, 800),
    TweetGen()
]

GEN_TRAIN_SET = [
    SinGen(500, 700, 200), 
   # RampGen(10, 800),
   # StepGen(range(0, 1000, 100), range(0, 10000, 1000)),
   # StepGen([50, 800, 1000], [50, 5000, 50]),

]

GEN_SET_test = [
    SinGen(150, 160, 200), 
    StepGen([400,800,1200,1600], [50, 300,50,300]),
    RampGen(slope=10, steady=50, initial=200, rampstart=10),
    TweetGen(),
    RampGen(2, 300),
    SinGen(1000, 1100, 100),
    StepGen(range(0, 1000, 100), range(0, 10000, 1000)),
    StepGen([50, 800, 1000], [50, 5000, 50]),
    RampGen(10, 800),
    RampGen(20, 800)
]

# CONTROLLERS
CONTROLLER_SET_INDUSTRY = [
    StaticController(VM_PERIOD, INIT_CORES, "Static (1)"),
    RBControllerWithCooldown(VM_PERIOD, INIT_CORES, step=1, cooldown=0, name="SimpleVM"),
    RBControllerWithCooldown(CONTAINER_PERIOD, INIT_CORES, step=1, cooldown=0, name="SimpleCR"),
    RBControllerWithCooldown(VM_PERIOD, INIT_CORES, step=3, cooldown=0, name="SimpleVM +3"),
    RBControllerWithCooldown(CONTAINER_PERIOD, INIT_CORES, step=3, cooldown=0, name="SimpleCR +3"),
    StepController(VM_PERIOD, INIT_CORES, {APP_SLA*0.8: 0.9, APP_SLA*0.9: 1, APP_SLA: 1.1, APP_SLA*1.1: 1.2, APP_SLA*1.201: 1.3}, cooldown=0, name="StepVM"),
    StepController(CONTAINER_PERIOD, INIT_CORES, {APP_SLA*0.8: 0.9, APP_SLA*0.9: 1, APP_SLA: 1.1, APP_SLA*1.1: 1.2, APP_SLA*1.201: 1.3}, cooldown=0, name="StepCR"),
    TargetController(VM_PERIOD, INIT_CORES, cooldown=0, name="TargetVM"),
    TargetController(CONTAINER_PERIOD, INIT_CORES, cooldown=0, name="TargetCR"),
    TargetController(SCALEX_PERIOD, INIT_CORES, cooldown=0, name="TargetFast")
]

SCALEX = CTControllerScaleX(SCALEX_PERIOD, INIT_CORES, min_cores=MIN_CORES, max_cores=MAX_CORES, st=SET_POINT_FACTOR, name="ScaleX")
OPT = OPTCTRL(OPTCTRL_PERIOD, init_cores=INIT_CORES, st=SET_POINT_FACTOR, min_cores=MIN_CORES, max_cores=MAX_CORES,name="QNCTRL")
ROBUST = OPTCTRLROBUST(OPTCTRL_PERIOD, init_cores=INIT_CORES, st=SET_POINT_FACTOR, min_cores=MIN_CORES, max_cores=MAX_CORES,name="QNCTRLROBUST")
JOINT = JointController(OPTCTRL_PERIOD, init_cores=INIT_CORES, min_cores=MIN_CORES, max_cores=MAX_CORES,st=SET_POINT_FACTOR, name="Joint")
RL = RLController(SCALEX_PERIOD, INIT_CORES, MIN_CORES, MAX_CORES, SET_POINT_FACTOR, "RLController")

PPO = PPOController(
    SCALEX_PERIOD, INIT_CORES,
    min_cores=MIN_CORES, max_cores=MAX_CORES,
    st=SET_POINT_FACTOR,
    name="PPOController"
)

PPO_GUARD = PPOController(
    SCALEX_PERIOD, INIT_CORES,
    min_cores=MIN_CORES, max_cores=MAX_CORES,
    st=SET_POINT_FACTOR, name="PPO-Guard",
    burst_mode="guard",          # <-- nuovo
    burst_threshold_q=20,        # salto coda
    burst_threshold_r=30,        # salto req/s
    burst_extra=4
)

PPO_HYBRID = PPOController(
    SCALEX_PERIOD, INIT_CORES,
    min_cores=MIN_CORES, max_cores=MAX_CORES,
    st=SET_POINT_FACTOR, name="PPO-Hybrid",
    burst_mode="hybrid",         # RL + guard-rail
    burst_threshold_q=15,
    burst_threshold_r=25,
    burst_extra=3,
    trend_features=True          # aggiunge queue_delta e rate_delta
)

# APPS
APPLICATION_1 = Application1(sla=APP_SLA, init_cores=INIT_CORES)
APPLICATION_Noisy = Application1Noisy(sla=APP_SLA, init_cores=INIT_CORES)
APPLICATION_2 = ApplicationMVA(sla=APP_SLA,stime=APP_2_S_TIME,init_cores=INIT_CORES)
APPLICATION_MMC = applicationMMC(sla=APP_SLA,stime=APP_MMC_S_TIME,init_cores=INIT_CORES)