import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import array


# CV
def CV(spike_train):
    interspike_intervals = np.diff(spike_train)
    isi_mean = np.mean(interspike_intervals)
    isi_std = np.std(interspike_intervals)
    CV = isi_std / isi_mean

    return CV


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

def Fano(counts):
    mean = np.mean(counts)
    var = np.var(counts)
    Fano = var / mean


    return Fano


neuron_A = np.loadtxt('neuron_A.csv', delimiter=',')

neuron_A_CV = CV(neuron_A)
print('neuron_A_CV is:', neuron_A_CV)



for bin in([100, 300, 600, 1000]):
    spike_count = cal_counts(neuron_A, bin)
    fano = Fano(spike_count)
    print('bin', bin, 'neuron_A_Fano is:', fano)
    print('\n')
