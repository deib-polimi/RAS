from generators import *
from controllers import *
from runner import Runner
from applications import SockShopFunction

appSLA = 0.6
horizon = 1000
monitoringWindow = 10
initCores = 1

scaleXPeriod = 1

c = CTControllerScaleX(scaleXPeriod, initCores)
c.setName("ScaleX")

cartsDelete = SockShopFunction(appSLA, a1=0.00064, a2=0.00006636302, a3=0.00073)
orders = SockShopFunction(appSLA, a1=0.00061, a2=0.00021140362, a3=0.00018)
orders.cores = 1.1148
print(orders.setRT(100))


runner = Runner(horizon, [c], monitoringWindow, orders)

g = RampGen(2, 45, 10)
g.setName("RMP1")
runner.run(g)

runner.log()
runner.plot()
