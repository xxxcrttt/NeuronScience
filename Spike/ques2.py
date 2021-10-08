import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
import array
import csv

def cal_counts(spike_train, bin):
    end = bin
    spike_count = 0
    counts = []
    for spike in spike_train:
        if spike >= end:
            counts.append(spike_count)
            spike_count = 1
            diff = int((spike - end)/bin)
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

def cal_trial_counts(trial_id_set, spike_train, bin):
    end = bin
    id0_count = []
    id1_count = []
    spike_count = 0
    for spike in spike_train:
        if spike >= end:
            diff = int((spike - end) / 1000)
            last_trial = int(end/1000) - 1
            last_id = trial_id_set[last_trial]
            id1_count.append(spike_count) if last_id == 1 else id0_count.append(spike_count)
            spike_count = 1
            if diff > 0:
                for i in range(0, diff):
                    last_trial += 1
                    last_id = trial_id_set[last_trial]
                    id1_count.append(0) if last_id == 1 else id0_count.append(0)
                end = end + (diff + 1) * bin
            elif diff == 0:
                end += bin
        else:
            spike_count += 1

    return id0_count, id1_count

# plot histogram
def plt_hist(id0_count, id1_count):
    id0_maxbin = max(id0_count)
    id1_maxbin = max(id1_count)
    final_bin = max(id0_maxbin, id1_maxbin)

    n, bins, patches = plt.hist([id0_count, id1_count],
    bins = final_bin, color = ['blue', 'orange'],
    label = ['stimulus off', 'stimulus on'], rwidth = 1000)

    plt.legend()
    plt.title('Figure 1. spike count for the stimulus off and on')
    plt.xlabel('spike count (time/1s)')
    plt.ylabel('number of trial')
    plt.show()

    return n, bins

# d prime
def cal_d_prime(n):
    plus = n[1]
    minus = n[0]
    miu_plus = np.mean(plus)
    miu_minus = np.mean(minus)
    sigma_plus = np.std(plus)
    sigma_minus = np.std(minus)
    d_prime = (miu_plus - miu_minus) / sigma_plus

    return d_prime

# decoding
def decoding(spike_count, decision_boundary):
    if spike_count > decision_boundary:
        detection = 1
    elif spike_count < decision_boundary:
        detection = 0
    elif spike_count == decision_boundary:
        detection = 2
    return detection

# rates
def cal_rates(neuron_A, trial_id_set, decision_boundary, total_posi, total_nega):
    posi = 0
    nega = 0
    total_correct = 0
    if_flag = 0
    counts = cal_counts(neuron_A, 1000)
    for i in range(0, 1000):
        spike_count = counts[i]
        true_id = trial_id_set[i]
        detection = decoding(spike_count, decision_boundary)
        if detection == true_id:
            total_correct += 1
            if detection == 1:
                posi += 1
            if detection == 0:
                nega += 1
    tpr = posi / total_posi
    tnr = nega / total_nega
    tcr = total_correct / 1000
    return tpr, tnr, tcr

# plot
def plot_rates(p_rates, n_rates, total_rates):
    xs = np.arange(0, 40, 1)
    plt.plot(xs, p_rates, marker = 'o', ms = 3, label = 'true positive rates', color = 'blue')
    plt.plot(xs, n_rates, marker = 'o', ms = 3, label = 'true negative rates', color = 'orange')
    plt.plot(xs, total_rates, marker = 'o', ms = 3, label = 'total correct rates')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    plt.title('Figure 2. Overlaying three curves')
    plt.xlabel('decision_boundary')
    plt.ylabel('correct rates(%)')
    plt.legend()
    plt.show()


def main():
    neuron_A = np.loadtxt('neuron_A.csv', delimiter = ',')
    trial_id_set = np.loadtxt('trial_ID.csv', delimiter = ',')

    # Q2-1.
    id0_count, id1_count = cal_trial_counts(trial_id_set, neuron_A, 1000)
    total_posi = len(id1_count)
    total_nega = len(id0_count)

    n, bins = plt_hist(id0_count, id1_count)
    d_prime = cal_d_prime(n)
    print('d_prime is:', d_prime)

    #========================================================
    # Q2-2.
    p_rates = []
    n_rates = []
    total_rates = []
    for i in range(0, 40):
        p_rate, n_rate, total_rate = cal_rates(neuron_A, trial_id_set, i, total_posi, total_nega)
        p_rates.append(p_rate)
        n_rates.append(n_rate)
        total_rates.append(total_rate)
    plot_rates(p_rates, n_rates, total_rates)

main()
