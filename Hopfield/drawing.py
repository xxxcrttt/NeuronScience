import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import csv
import array


train = np.loadtxt('train_images.csv', delimiter = ',')
test = np.loadtxt('test_images.csv', delimiter =',')

test = test[1].reshape(28,28)
print(test.shape)

plt.imshow(test)
plt.show()
