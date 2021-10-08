import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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


# plot histtogram
def plt_hist(id0_count, id1_count):
    id0_maxbin = max(id0_count)
    id1_maxbin = max(id1_count)
    final_bin = max(id0_maxbin, id1_maxbin)

    n, bins, patches = plt.hist([id0_count, id1_count],
    bins = final_bin, color = ['blue', 'orange'],
    label = ['stimulus off', 'stimulus on'], rwidth = 1000)
    plt.legend()
    plt.title('Figure 3. Neuron_B stimulus on and off')
    plt.xlabel('spike count (times/1s)')
    plt.ylabel('number of trial')
    plt.show()
    return n, bins


# d_prime
def cal_d_prime(n):
    plus = n[1]
    minus = n[0]
    miu_plus = np.mean(plus)
    miu_minus = np.mean(minus)
    sigma_plus = np.std(plus)
    sigma_minus = np.std(minus)
    d_prime = (miu_plus - miu_minus) / sigma_plus
    return d_prime

# joint decoding
def joint_decoding(xA, xB):
    if xA - xB - 6 > 0:
        prediction = 1
    else:
        prediction = 0

    return prediction

# joint Rates
def joint_get_rates(A_spikes, B_spikes, trial_id_set, total_posi, total_nega):
    posi = 0
    nega = 0
    total_correct = 0
    xAs = cal_counts(A_spikes, 1000)
    xBs = cal_counts(B_spikes, 1000)
    for i in range(0, 1000):
        xA = xAs[i]
        xB = xBs[i]
        true_id = trial_id_set[i]
        prediction = joint_decoding(xA, xB)
        if prediction == true_id:
            total_correct += 1
            if prediction == 1:
                posi += 1
            if prediction == 0:
                nega += 1
    tpr = posi / total_posi
    tnr = nega / total_nega
    total_correct_rate = total_correct / 1000

    return tpr, tnr, total_correct_rate

# decoding
def decoding(spike_count, decision_boundary):
    if spike_count > decision_boundary:
        prediction = 1
    elif spike_count < decision_boundary:
        prediction = 0
    elif spike_count == decision_boundary:
        prediction = 2
    return prediction

# get_rates
def get_rates(neuron_A, trial_id_set, decision_boundary, total_posi, total_nega):
    posi = 0
    nega = 0
    total_correct = 0
    id_flag = 0
    counts = cal_counts(neuron_A, 1000)
    for i in range(0, 1000):
        spike_count = counts[i];
        true_id = trial_id_set[i]
        prediction = decoding(spike_count, decision_boundary)
        if prediction == true_id:
            total_correct += 1
            if prediction == 1:
                posi += 1
            if prediction == 0:
                nega += 1
    tpr = posi / total_posi
    tnr = nega / total_nega
    total_correct_rate = total_correct / 1000

    return tpr, tnr, total_correct_rate

# plot
def plot_rates(p_rates, n_rates, total_rates):
    xs = np.arange(0, 40, 1)
    plt.plot(xs, p_rates, marker ='o', ms = 3, label ='true positive rates', color = 'red')
    plt.plot(xs, n_rates, marker ='o', ms = 3, label ='true negative rates', color = 'green')
    plt.plot(xs, total_rates, marker ='o', ms = 3, label = 'total correct rate', color = 'blue')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    plt.title('Figure 4. Plot of Correct Rates')
    plt.xlabel('decision_boundary')
    plt.ylabel('correct rates(%)')
    plt.legend()
    plt.show()

# ===== main ================

def main():
    neuron_A = np.loadtxt('neuron_A.csv', delimiter =',')
    neuron_B = np.loadtxt('neuron_B.csv', delimiter =',')
    trial_id_set = np.loadtxt('trial_ID.csv', delimiter =',')


# Q3-1. ==================
    id0_count, id1_count = cal_trial_counts(trial_id_set, neuron_A, 1000)
    B_total_posi = len(id1_count)
    B_total_nega = len(id0_count)
    #print(id1_count)

    n, bins = plt_hist(id0_count, id1_count)
    d_prime = cal_d_prime(n)
    print(d_prime)

# ploting =============
    p_rates = []
    n_rates = []
    total_rates = []
    for i in range(0, 40):
        p_rate, n_rate, total_rate = get_rates(neuron_B, trial_id_set, i, B_total_posi, B_total_nega)
        p_rates.append(p_rate)
        n_rates.append(n_rate)
        total_rates.append(total_rate)
    plot_rates(p_rates, n_rates, total_rates)


# Q3-2. Joint Rule ================
    joint_p_rate, joint_n_rate, joint_total_rate = joint_get_rates(neuron_A, neuron_B, trial_id_set, B_total_posi, B_total_nega)
    print('joint positive rate:', joint_p_rate)
    print('joint negative rate:', joint_n_rate)
    print('joint total rate:', joint_total_rate)


main()
