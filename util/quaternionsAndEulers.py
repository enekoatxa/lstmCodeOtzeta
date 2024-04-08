from scipy.spatial.transform import Rotation as R 
import numpy as np
import timeit

##################################################################
# global variables for glueing and separating quaternion vectors #
##################################################################
numberOfRotationValues = 0
numberOfPositionValues = 0

#####################################
# conversion quaternion <---> euler #
#####################################

def fromEulerToQuaternion(eulerRotation):
    quat_values = eulerRotation.as_quat()
    r_quat = R.from_quat(np.array([quat_values[0], quat_values[1], quat_values[2], quat_values[3]]))
    return r_quat

def fromQuaternionToEuler(quaternionRotation):
    euler_values = quaternionRotation.as_euler('xyz', degrees = True)
    r_euler = R.from_euler('xyz', np.array([euler_values[0], euler_values[1], euler_values[2]]), degrees=True)
    return r_euler

def fromEulerToQuaternionValues(r, p, y):
    r_euler = R.from_euler('xyz', np.array([r, p, y]), degrees=True)
    quat_values = r_euler.as_quat()
    return quat_values[0], quat_values[1], quat_values[2], quat_values[3]


def fromQuaternionToEulerValues(w, x, y, z):
    r_quat = R.from_quat(np.array([w, x, y, z]))
    euler_values = r_quat.as_euler('xyz', degrees = True)
    print(euler_values)
    return euler_values[0], euler_values[1], euler_values[2]

def fromEulerToQuaternionVector(eulerVector):
    ret = []
    for index in range(0, len(eulerVector)):
        if(index%3==0):
            w, x, y, z = fromEulerToQuaternionValues(eulerVector[index], eulerVector[index+1], eulerVector[index+2])
            ret.append(w)
            ret.append(x)
            ret.append(y)
            ret.append(z)
    return ret

def fromQuaternionToEulerVector(quaternionVector):
    ret = []
    for index in range(0, len(quaternionVector)):
        if(index%4==0):
            x, y, z = fromQuaternionToEulerValues(quaternionVector[index], quaternionVector[index+1], quaternionVector[index+2], quaternionVector[index+3])
            ret.append(x)
            ret.append(y)
            ret.append(z)
    return ret


#######################################################
# glueing and separating vectors to position/rotation #
#######################################################

def calculatePositionIndexesFromHeader(path):
    currentIndex = 0
    positionIndexes = []
    rotationIndexes = []
    f = open(path, "r")
    line = f.readline()
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

def separateVector(wholeVector, path="/home/bee/Desktop/idle animation generator/enekoDataset/genea2023_trn/idle_0000.bvh"):
    positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader(path)
    rotationsVector = []
    positionsVector = []
    for index in range(0, len(wholeVector)):
        if(index in rotationIndexes):
            rotationsVector.append(wholeVector[index])
        else:
            positionsVector.append(wholeVector[index])
    return rotationsVector, positionsVector

def glueVectors(rotationsVector, positionsVector, path="/home/bee/Desktop/idle animation generator/enekoDataset/genea2023_trn/idle_0000.bvh"):
    positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader(path)
    wholeVector = [None] * (len(positionIndexes) + len(rotationIndexes))
    for index in range(0, len(rotationsVector)):
        wholeVector[rotationIndexes[index]] = rotationsVector[index]
    for index in range(0, len(positionsVector)):
        wholeVector[positionIndexes[index]] = positionsVector[index]
    return wholeVector

# this is faster than list comprehensions (tested with timeit)
def concatenateVectorsSimple(rotationsVector, positionsVector):
    wholeVector = []
    for rotation in rotationsVector:
        wholeVector.append(rotation)
    for position in positionsVector:
        wholeVector.append(position)
    # change the global variables, so we know where to separate later
    global numberOfRotationValues
    numberOfRotationValues = len(rotationsVector)
    return wholeVector

def separateVectorsSimple(wholeSimpleVector):
    rotationsVector = []
    positionsVector = []
    for index in range(0, len(wholeSimpleVector)):
        if(index<numberOfRotationValues):
            rotationsVector.append(wholeSimpleVector[index])
        else:
            positionsVector.append(wholeSimpleVector[index])
    return rotationsVector, positionsVector