import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

def dV(Vm, I, timestep):
    Vrest = -72e-3
    Vth = -52e-3
    Vreset = -62e-3
    Rm = 100e6
    tau_m = 30e-3
    #El = -72e-3
    #Ie = 0.21e-9
    #dt = 0.1e-3

    if Vm >= Vth:
        Vm = Vreset
    else:
        Vm = Vm + timestep * 1/tau_m *((Vrest - Vm) + Rm * I)
    return Vm


def main():

    # Q1. LIF for 1s
    I = 0.21e-9
    timestep = 0.1e-3
    time = np.arange(0, 1+timestep, timestep)
    Vms = np.zeros(np.shape(time)[0])
    Vms[0] = random.uniform(-72e-3, -52e-3)
    flag = 0

    for item in time[0:-1]:
        Vms[flag + 1] = dV(Vms[flag], I, timestep)
        flag += 1

    plt.plot()
    plt.plot(time, Vms)
    plt.title('Change of V with injected current = 0.21nA')
    plt.xlabel("time /s")
    plt.ylabel("Memberance Potential /V")
    plt.legend()
    plt.show()



    # Q1-2. F-I curve
    I_list = np.arange(0, 0.5e-9, 0.5e-9/10.)
    F_list = np.zeros(np.shape(I_list))
    I_flag = 0
    for i in I_list:
        I = i
        spike_count = 0

        timestep = 0.1e-3
        list = np.arange(0, 1+timestep, timestep)
        Vms = np.zeros(np.shape(list)[0])
        Vms[0] = -72e-3
        flag = 0

        for item in list[0:-1]:
            Vms[flag + 1] = dV(Vms[flag], I, timestep)
            if Vms[flag + 1] >= -52e-3:
                spike_count += 1
            flag += 1

        F_list[I_flag] = spike_count
        I_flag += 1
        #print(I,F_list)

        plt.plot(I_list, F_list)
        plt.title('F_I curve for the current varies from 0-0.5nA.')
        plt.xlabel('Input current /A')
        plt.ylabel('Frequence /Hz')
        plt.show()

main()
