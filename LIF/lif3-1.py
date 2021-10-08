import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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
gi = 0.5 * 10 ** -9 #nanoSiemens (A value of conductance)

#r1
numSynapses = 50
avgr1 = 10
avgr2 = 100
B1 = 5
B2 = 0
f = 2

gBarArray1 =[gi] * 50
gBarArray2 =[gi] * 50
sArray1 = []
sArray2 = []

sArray1 = np.zeros((300000, 50))
sArray2 = np.zeros((300000, 50))

t1 = np.arange(0, 30-dt, dt)
nVValues1 = [Vrest]

t2 = np.arange(0, 30, dt)
nVValues2 = [Vrest]


nVm1 = []
nVm = Vrest
numSpikes1 = 0

sVPrev = Vrest
input1spike = 0
input2spike = 0

for i in range(1, 300000):

    synapseVoltageTally1 = 0
    synapseVoltageTally2 = 0


    #nVPrev1 = nVValues1[i-1]
    #nVPrev2 = nVValues2[i-1]
    r1t = (avgr1 + B1 * np.sin(2 * np.pi * f * i * dt)) * dt
    r2t = (avgr2 + B2 * np.sin(2 * np.pi * f * i * dt)) * dt


    for synapseNum1 in range(50):


        sPrev1 = sArray1[i-1][synapseNum1]
        sPrev2 = sArray2[i-1][synapseNum1]

        if random.random() < r1t:
            sV1 = sPrev1 - sPrev1 *  dt/Ts + deltaS
            input1spike += 1
        else:
            sV1 = sPrev1 - sPrev1 *  dt/Ts

        if random.random() < r2t:
            sV2 = sPrev2 - sPrev2 *  dt/Ts + deltaS
            input2spike += 1
        else:
            sV2 = sPrev2 - sPrev2 *  dt/Ts

        #sArray1[synapseNum1].append(sV1)
        #sArray2[synapseNum1].append(sV2)
        sArray1[i][synapseNum1] = sV1
        sArray2[i][synapseNum1] = sV2

        synapseVoltageTally1 += (gBarArray1[synapseNum1] * sV1)
        synapseVoltageTally2 += (gBarArray2[synapseNum1] * sV2)

    RmIs1 = Rm * synapseVoltageTally1 /50
    RmIs2 = Rm * synapseVoltageTally2 /50

    dnv  = ((Vrest - sVPrev) + RmIs1 + RmIs2) * (dt/Tm)

    nVm  = sVPrev + dnv

    sVPrev = nVm

    if nVm >= Vth:
        nVm = Vrest
        sVPrev = Vrest
        numSpikes1 += 1



    nVm1.append(nVm)
        #print(nVm)

print(input1spike+input2spike)

print(numSpikes1)

# #
# plt.plot(t1, nVm1,label="B1=0Hz, B2=50Hz")
# plt.title("Voltage as a function of time.")
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.legend()
# plt.show()
