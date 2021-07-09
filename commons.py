from generators import *

SN1 = SinGen(500, 700, 200)
SN1.setName("SN1")

SN2 = SinGen(1000, 1000, 1500)
SN2.setName("SN2")

SP1 = StepGen(range(0, 1000, 100), range(0, 10000, 1000))
SP1.setName("SP1")


SP2 = StepGen([50, 800, 1000], [50, 30000, 50])
SP2.setName("SP2")


RP1 = RampGen(10, 800)
RP1.setName("RP1")

RP2 = RampGen(20, 800)
RP2.setName("RP2")


ALL = [SN1, SN2, SP1, SP2, RP1, RP2]
