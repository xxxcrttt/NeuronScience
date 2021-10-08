import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

# Coefficient of Variation
def CV(spike_train):
    interspike_intervals = np.diff(spike_train)
    isi_mean = np.mean(interspike_intervals)
    isi_std = np.std(interspike_intervals)
    CV = isi_std / isi_mean

    return CV

# count for spike_count
def cal_counts(spike_train,bin):
    end = bin
    spike_count = 0
    counts = []
    for spike in spike_train:
        if spike >= end:
            counts.append(spike_count)
            spike_count = 1
            diff = int((spike - end) / bin)
            if diff > 0:
                end = end + (diff + 1) * bin
                for i in range(0, diff):
                    counts.append(0)
            elif diff == 0:
                end += bin
                if spike == spike_train[-1]:
                    counts.append(spike_count)
        else:
            spike_count += 1
            if spike == spike_train[-1]:
                counts.append(spike_count)
    return counts

#Fano
def cal_Fano(counts):
    mean = np.mean(counts)
    var = np.var(counts)
    Fano = var / mean

    return Fano

# Trial_ID cv
def cal_trial_CV(trial_set, spike_set):
    end = 1000
    id0_interval = []
    id1_interval = []
    for i in range (0, len(spike_set)-1):
        trial_id = trial_set[int(end / 1000 - 1)]
        if spike_set[i] < end and spike_set[i+1] < end:
            if trial_id == 0:
                id0_interval.append(spike_set[i+1] - spike_set[i])
            elif trial_id == 1:
                id1_interval.append(spike_set[i+1] - spike_set[i])
        elif spike_set[i] < end and spike_set[i+1] >= end:
            continue
        elif spike_set[i] >= end and spike_set[i+1] >= end:
            end += 1000
            if trial_id == 0:
                id0_interval.append(spike_set[i+1] - spike_set[i])
            elif trial_id == 1:
                id1_interval.append(spike_set[i+1] - spike_set[i])
    id0_std = np.std(id0_interval)
    id0_mean = np.mean(id0_interval)
    id1_std = np.std(id1_interval)
    id1_mean = np.mean(id1_interval)
    id0_cv = id0_std/id0_mean
    id1_cv = id1_std/id1_mean
    #print(id1_interval)

    return id0_cv, id1_cv

# fano
def cal_trial_fano(trial_id_set, spike_time_set, bin):
    end = bin
    id1_spike_count = []
    id0_spike_count = []
    spike_count = 0
    for spike in spike_time_set:
        flag = int((end - bin)/ 1000)
        id = trial_id_set[flag]
        if spike < end:
            spike_count += 1
        elif spike >= end:
            while spike >= end:
                flag = int((end - bin)/ 1000)
                id = trial_id_set[flag]
                if id == 0:
                    id0_spike_count.append(spike_count)
                    spike_count = 0
                elif id == 1:
                    id1_spike_count.append(spike_count)
                    spike_count = 0
                end += bin
            spike_count = 1
        if spike == spike_time_set[-1]:
            id = trial_id_set[-1]
            id1_spike_count.append(spike_count) if id == 1 else id0_spike_count.append(spike_count)


    id1_fano = cal_Fano(id1_spike_count)
    id0_fano = cal_Fano(id0_spike_count)

    return id0_fano, id1_fano

def main():
    neuron_A = np.loadtxt('neuron_A.csv', delimiter = ',')
    trial_id_set = np.loadtxt('trial_ID.csv', delimiter=',')

    id0_cv, id1_cv = cal_trial_CV(trial_id_set, neuron_A)
    print('trial_0_cv is:', id0_cv)
    print('trial_1_cv is:', id1_cv)


    for bin in ([100, 300, 600, 1000]):
        spike_count = cal_counts(neuron_A, bin)
        fano = cal_Fano(spike_count)
        id0_fano, id1_fano = cal_trial_fano(trial_id_set, neuron_A, bin)
        print('bin', bin, 'id0_fano is:', id0_fano)
        print('bin', bin, 'id1_fano is:', id1_fano)
        print('\n')

main()
