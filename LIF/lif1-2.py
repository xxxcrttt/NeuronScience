import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

Vrest = -72e-3
Vth = -52e-3
Vreset = -62e-3
Rm = 100e6
tau_m = 30e-3
El = -72e-3
#Ie = 0.21e-9
dt = 0.1e-3
I_array = np.arange(0, 0.5e-9, 0.5e-9/10.)

def dV(Vm, I, dt):
    if Vm >= Vth:
        Vm = Vreset
    else:
        Vm = Vm + dt * 1/tau_m *((Vrest - Vm) + Rm * I)

    return Vm



def main():
    # 10s, Ie between 0 ~ 0.5

    I_array = np.arange(0, 0.5e-9, 0.5e-9/100.)
    F_array = np.zeros(np.shape(I_array))
    I_flag = 0

    for i in I_array:
        I = i
        spike_count = 0

        dt = 0.1e-3
        time = np.arange(0, 10+dt, dt)
        Vms = np.zeros(np.shape(time)[0])
        Vms[0] = -72e-3
        flag = 0

        for item in time[0:-1]:
            Vms[flag + 1] = dV(Vms[flag],I,dt)
            if Vms[flag + 1] >= -52e-3:
                spike_count += 1
            flag += 1

        F_array[I_flag] = spike_count
        I_flag += 1

        plt.plot(I_array, F_array)
        plt.title('F-I curve for the current varies from 0-0.5nA')
        plt.xlabel('Input Current / A')
        plt.ylabel('Frequency / HZ')
        plt.show()

main()
