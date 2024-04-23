import numpy as np
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
from statistics import variance
import matplotlib.pyplot as plt

# mean and variance of each element in the 498 dim vector
data, ids = bvhLoader.loadDataset("silenceDataset3sec360", "Validation", specificSize=100, trim=True, verbose=True)
data = np.asarray(data)
counter = 0
variances = []
means = []
indexes = np.arange(0,497)
for person in data:
    person = np.transpose(person)
    for element in person:
        counter+=1
        print(str(counter) + ":::" + str(np.var(element)))
        if(np.var(element)>100):
            print("############")
            print(element)
            print("############")
        variances.append(np.var(element, dtype=np.float64))
        means.append(np.mean(element, dtype=np.float64))
        
    fig, ax = plt.subplots()
    ax.plot(indexes, variances)
    # ax.plot(indexes, means)
    plt.show()
    variances.clear()
    means.clear()
    counter=0
print(data.shape)