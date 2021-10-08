import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dt = 0.1 * 10 ** -3 #ms -- timestep
El = -72 * 10 ** -3 #mV
Tm = 30 * 10 ** -3 #ms
Rm = 100 * 10 ** 6 #MOh
Vrest = -72 * 10 ** -3 #mV
Vth = -52 * 10 ** -3 # mV
Vreset = -62e-3

Ts = 2 * 10 ** -3
Es = 0
deltaS = 1.0
initialgBar = 0.5 * 10 ** -9 #nanoSiemens (A value of conductance)

# group 1
numSynapses1 = 50
firingRate1 = 10

# group 2
numSynapses2 = 50
firingRate2 = 100

gBarArray1 =[initialgBar] * 50
gBarArray2 =[initialgBar] * 50
sArray1 = []
sArray2 = []
for x in range(0, 100):
    sArray1.append([0])
    sArray2.append([0])
    RmIsArray = [0] * 50

tValues1 = np.arange(0, 1-dt, dt)
nVValues1 = [Vrest]

tValues2 = np.arange(0, 1, dt)
nVValues2 = [Vrest]

numSpikes1 = 0
numSpikes2 = 0

nVm1 = []
nVm = Vrest

sArray1 = np.zeros((10000, 50))
sArray2 = np.zeros((10000, 50))

sVPrev = Vrest


for i in range(1, 10000):
    synapseVoltageTally1 = 0
    synapseVoltageTally2 = 0


    #nVPrev1 = nVValues1[i-1]
    #nVPrev2 = nVValues2[i-1]

    for synapseNum1 in range(50):

        sPrev1 = sArray1[i-1][synapseNum1]
        sPrev2 = sArray2[i-1][synapseNum1]

        if random.random() < firingRate1 * dt:
            sV1 = sPrev1 - sPrev1 *  dt/Ts + deltaS
        else:
            sV1 = sPrev1 - sPrev1 *  dt/Ts

        if random.random() < firingRate2 * dt:
            sV2 = sPrev2 - sPrev2 *  dt/Ts + deltaS
        else:
            sV2 = sPrev2 - sPrev2 *  dt/Ts

        #sArray1[synapseNum1].append(sV1)
        #sArray2[synapseNum1].append(sV2)
        sArray1[i][synapseNum1] = sV1
        sArray2[i][synapseNum1] = sV2

        synapseVoltageTally1 += (gBarArray1[synapseNum1] * sV1)
        synapseVoltageTally2 += (gBarArray2[synapseNum1] * sV2)

    RmIs1 = Rm * synapseVoltageTally1 / 50
    RmIs2 = Rm * synapseVoltageTally2 / 50

    dnv  = ((Vrest - sVPrev) + RmIs1 + RmIs2) * (dt/Tm)

    nVm  = sVPrev + dnv

    sVPrev = nVm


    if nVm >= Vth:
        nVm = Vrest
        sVPrev = Vrest
        numSpikes1 += 1

    nVm1.append(nVm)
        #print(nVm)

print(numSpikes1)



plt.plot(tValues1, nVm1)
plt.title("Neuron with 100 Synapses mixed together")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.show()
