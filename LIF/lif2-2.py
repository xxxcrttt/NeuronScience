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
gi = 0.5e-9 #nanoSiemens (A value of conductance)

# group 1
numSynapses1 = 50
firingRate1 = np.arange(0, 150, 10.)

numSynapses2 = 50
firingRate2 = 10

giBar1 =[gi] * 50
giBar2 =[gi] * 50
sArray1 = []
sArray2 = []

sArray1 = np.zeros((100000, 50))
sArray2 = np.zeros((100000, 50))

tValues1 = np.arange(0, 10-dt, dt)
nVValues1 = [Vrest]


tValues2 = np.arange(0, 10, dt)
nVValues2 = [Vrest]

input1 = np.arange(0, 150, 10.)
output1 = np.zeros(np.shape(input1))

sVPrev = Vrest
nVm1 = []


for i in range (15):
    numSpikes1 = 0
    #numSpikes2 = 0
    nVm = Vrest
    sVPrev = 0

    for j in range(100000):
        for synapseNum1 in range(50):
            sArray1[j][synapseNum1] = 0;
            sArray2[j][synapseNum1] = 0;


    for j in range(1, 100000):
        synapseVoltageTally1 = 0
        synapseVoltageTally2 = 0

        for synapseNum1 in range(50):

            sPrev1 = sArray1[j-1][synapseNum1]
            sPrev2 = sArray2[j-1][synapseNum1]

            if random.random() < firingRate1[i] * dt:
                sV1 = sPrev1 - sPrev1 *  dt/Ts + deltaS


            else:
                sV1 = sPrev1 - sPrev1 *  dt/Ts

            if random.random() < firingRate2 * dt:
                sV2 = sPrev2 - sPrev2 *  dt/Ts + deltaS

            else:
                sV2 = sPrev2 - sPrev2 *  dt/Ts

            #sArray1[synapseNum1].append(sV1)
            #sArray2[synapseNum1].append(sV2)
            sArray1[j][synapseNum1] = sV1
            sArray2[j][synapseNum1] = sV2

            synapseVoltageTally1 += (giBar1[synapseNum1] * sV1)
            synapseVoltageTally2 += (giBar2[synapseNum1] * sV2)

        RmIs1 = Rm * synapseVoltageTally1 / 50
        RmIs2 = Rm * synapseVoltageTally2 / 50

        dnv  = ((Vrest - sVPrev) + RmIs1 + RmIs2) * (dt/Tm)

        nVm  = sVPrev + dnv

        sVPrev = nVm

        if nVm >= Vth:
            nVm = Vreset
            sVPrev = Vreset
            numSpikes1 += 1

        nVm1.append(nVm)
    output1[i] = numSpikes1

print(output1)
print(numSpikes1)

plt.plot(input1, output1)
plt.title("Output spike frequence as a function of the input spike frequency.")
plt.xlabel("Firing rate varing from 0 to 150 (Hz)")
plt.ylabel("spike frequency (Hz)")
plt.legend()
plt.show()
