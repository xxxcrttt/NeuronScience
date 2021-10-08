import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import array
import csv
np.set_printoptions(suppress=True)


# 1. Create a weight matrix
def weight_matrix(a, b):
    l = len(a)
    weight = np.zeros([l,l])
    for i in range(0, l):
        for j in range(0, l):
            if i == j:
                weight[i, j] = 0
            else:
                weight[i, j] = a[i] * a[j] / b
                weight[j, i] = weight[i, j]

    return weight

# 3. evolve function for McCulloch-Pitts formula;
# 4. for two test pattern:
def evolve(test, weight_model, timestep, theta = 0):

    plt.imshow(np.reshape(test, (28, 28)))
    plt.title('evolve test_images_1: timestep = 0')
    plt.show()

    steps = []
    energys = []
    steps.append(0)
    energys.append(cal_energy(weight_model,test))

    for i in range(0, timestep):
        l = len(test)
        random_cell = random.randint(0, l - 1)
        nextstep = np.dot(weight_model[random_cell][:], test) - theta
        if nextstep > 0:
            test[random_cell] = 1
        elif nextstep < 0:
            test[random_cell] = -1
        energy = cal_energy(weight_model, test)
        steps.append(i)
        energys.append(energy)

        # if i == 999 or i = 2999, 0, 1000, 3000
        # 0, 800, 4000
        if i == 799 or i == 3999:
            plt.imshow(np.reshape(test, (28, 28)))
            title = 'evolve test images: timestep = ' + str(i + 1)
            plt.title(title)
            plt.show()

    return test, steps, energys

# Q2. energy
def cal_energy(weight_model, test):
    return -1/2 * test.dot(weight_model).dot(test.T)




# main ==============
def main():
    train = np.loadtxt('train_images.csv', delimiter =',')
    test = np.loadtxt('test_images.csv', delimiter =',')


    # 2. Fix the weight values
    weight_0 = weight_matrix(train[0], 1)
    weight_1 = weight_matrix(train[1], 1)
    weight_2 = weight_matrix(train[2], 1)

    weight_model = weight_0 + weight_1 + weight_2
    # plot
    plt.imshow(weight_model)
    plt.title('weight matrix')
    plt.show()

    # Q2. energy & plots
    energy = cal_energy(weight_model, test)
    test_evloved, steps, energys = evolve(test[1], weight_model, 0)
    plt.plot(steps, energys)
    plt.title('change of Energy to timestep for test image_2.')
    plt.xlabel('timestep')
    plt.ylabel('energy')
    plt.legend()
    plt.show()



main()
