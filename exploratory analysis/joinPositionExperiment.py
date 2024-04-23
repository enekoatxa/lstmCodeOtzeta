import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import operator

def gluePositionAndRotation(position, rotation):
    ret = []
    for posIndex in range(0, len(position), 3):
        ret.append(position[posIndex])
        ret.append(position[posIndex+1])
        ret.append(position[posIndex+2])
        ret.append(rotation[posIndex])
        ret.append(rotation[posIndex+1])
        ret.append(rotation[posIndex+2])
    return ret

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
data = bvhLoader.loadBvhToList("../../enekoDataset/genea2023_trn/idle_0003.bvh", returnHeader = False, returnData = True, returnCounter = False, 
                  onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0)
positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader("../../enekoDataset/genea2023_trn/idle_0003.bvh")
# plt.title('joint rotations')
# plt.ylabel('value')
# plt.xlabel('epoch')
# # data = np.transpose(data)
# for vec in data:
#     plt.plot(np.arange(0, 1780), vec)
#     plt.legend(['train'], loc='upper left')
# plt.show()
# plt.close()
firstVector = data[0]
newPositions = []
for index in range(0, len(data)):
    appendingVector = data[index]
    for index in positionIndexes:
        appendingVector[index] = firstVector[index]
    newPositions.append(appendingVector)

# plt.title('joint rotations')
# plt.ylabel('value')
# plt.xlabel('epoch')
# # newPositions = np.transpose(newPositions)
# for vec in newPositions:
#     plt.plot(np.arange(0, 1780), vec)
#     plt.legend(['train'], loc='upper left')
# plt.show()
# plt.close()
with open("specialRotations.bvh", "w") as f:
        for line in newPositions:
            f.write(str(line).replace("[", "").replace("]", "").replace(",", "").replace("\n", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close