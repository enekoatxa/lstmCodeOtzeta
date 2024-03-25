import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
import math

def calculatePositionIndexesFromHeader(path):
    currentIndex = 0
    positionIndexes = []
    rotationIndexes = []
    f = open(path, "r")
    line = f.readline()
    # read the header until the line "Frame Time: 0.0333333"
    while line.split(" ")[0]!= "Frame":
        for element in line.split(" "):
            if("position" in element):
                positionIndexes.append(currentIndex)
                currentIndex +=1
            if("rotation" in element):
                rotationIndexes.append(currentIndex)
                currentIndex +=1
        line = f.readline()
    return positionIndexes, rotationIndexes

# x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="dataset", partition="Validation", verbose=True, sequenceSize = n_steps, specificSize=1, trim=False, removeHandsAndFace=True, scaler=scaler, loadDifferences = True, jump=jump)
data = bvhLoader.loadBvhToList("../../enekoDataset/genea2023_trn/idle_0005.bvh", returnHeader = False, returnData = True, returnCounter = False, 
                  onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0)

positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader("../../enekoDataset/genea2023_trn/idle_0005.bvh")

plt.title('joint positions')
plt.ylabel('value')
plt.xlabel('epoch')
index = 0
for vec in data:
    if(index in positionIndexes):
        plt.plot(np.arange(0, len(data[0])), vec, label=index)
        plt.legend(loc="upper left")
    index += 1
plt.show()
plt.close()

plt.title('joint rotations')
plt.ylabel('value')
plt.xlabel('epoch')
index = 0
for vec in data:
    if(index in rotationIndexes):
        plt.plot(np.arange(0, len(data[0])), vec, label=index)
        plt.legend(loc="upper left")
    index += 1
plt.show()
plt.close()