import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
import quaternionsAndEulers
from numpy.random import randint
import numpy as np
import math
from scipy.spatial.transform import Rotation as R 
from scipy.spatial.transform import Slerp
n_seq = 30
seq_length = 200

# try with different easing functions
def easeInBack(x):
    c1 = 1.70158
    c3 = c1 + 1
    ret = c3 * x * x * x - c1 * x * x
    return ret

def easeInOutBack(x):
    c1 = 1.70158
    c2 = c1 * 1.525
    if x < 0.5:
        ret = (math.pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
    else:
        ret = (math.pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2
    return ret

def interpolateSequence(vectorA, vectorB):
    # first we calculate how long the interpolation should be (if the difference between frames is big, we want a longer animation, and viceversa)
    """
    diffMagnitude = 1000
    animationLengthMultiplier = np.linalg.norm((vectorA - vectorB)) / diffMagnitude
    interpolationBaseDuration = 200
    interpolationSeqSize = int(animationLengthMultiplier * interpolationBaseDuration)
    """

    interpolationPosition = []
    interpolationRotation = []
    interpolation = []
    rotationA, positionA = quaternionsAndEulers.separateVector(vectorA)
    rotationB, positionB = quaternionsAndEulers.separateVector(vectorB)

    # convert to np array
    rotationA = np.array(rotationA)
    rotationB = np.array(rotationB)
    positionA = np.array(positionA)
    positionB = np.array(positionB)
    # convert to quaternions
    rotationA = quaternionsAndEulers.fromEulerToQuaternionVector(rotationA)
    rotationB = quaternionsAndEulers.fromEulerToQuaternionVector(rotationB)
    interpolationSeqSize = 40
    # first, interpolate the positions part
    for step in range(1, interpolationSeqSize):
        intermediate_position_vector = positionA + (positionB - positionA) * (step/interpolationSeqSize)
        interpolationPosition.append(intermediate_position_vector.copy())
        positionA = intermediate_position_vector
    # TODO: then, interpolate the rotations part, using slerp (quaternions need slerp for correct interpolation)
    # to interpolate between the two rotation vectors, we need to do it quaternion by quaternion
    timesForQuaternionInterpolation = [(step/interpolationSeqSize) for step in range(1, interpolationSeqSize)]
    for index in range(0, len(rotationA)-3, 4):
        r = R.from_quat([[rotationA[index], rotationA[index+1], rotationA[index+2], rotationA[index+3]], [rotationB[index], rotationB[index+1], rotationB[index+2], rotationB[index+3]]])        
        slerp = Slerp(times=[0, 1], rotations=r)
        # create the interpolations and convert them to euler angles
        interpolations = slerp(timesForQuaternionInterpolation).as_euler("xyz", degrees=True)
        interpolations = np.transpose(interpolations)
        # put all the interpolations in a vector again
        for inter in interpolations:
            interpolationRotation.append(inter)
    interpolationRotation = np.transpose(interpolationRotation)
    rA = R.from_quat([rotationA[0], rotationA[0+1], rotationA[0+2], rotationA[0+3]])
    rB = R.from_quat([rotationB[0], rotationB[0+1], rotationB[0+2], rotationB[0+3]])

    # for sequence in range(1, len(interpolationRotation)):
    #     reshapedInterpolation = np.concatenate((reshapedInterpolation, interpolationRotation[sequence]), axis=1)
    # lastly, rejoin every vectors rotation and position
    for position, rotation in zip(interpolationPosition, interpolationRotation):
        interpolation.append(quaternionsAndEulers.glueVectors(rotationsVector = rotation, positionsVector = position))
    return np.asarray(interpolation)

def main():
    # load the dataset
    x, firstPersonFrames, ids = bvhLoader.loadDataset(datasetName="enekoDataset", partition="All", trim=False,
                                                verbose=True, onlyPositions=False,
                                                onlyRotations=False, removeHandsAndFace=False, 
                                                loadDifferences = False, jump = 0, specificSize = 1, useQuaternions = False)
    # load the header of a BVH, to write later automatically
    header = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/enekoDataset/genea2023_trn/idle_0000.bvh", returnHeader = True, returnData = False, returnCounter = False, 
                  onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0)
    # select random sequences
    allSequences = []
    for index in range (0, n_seq):
        personId = randint(0, len(x))
        seqId = randint(0, len(x[personId])-seq_length)
        # print("selected animation: " + str(personId) + "_" + str(seqId))
        allSequences.append(x[personId][seqId:seqId+seq_length])
    
    interpolations = []
    # create interpolators
    for index in range(0, len(allSequences)-1):
        interpolations.append(interpolateSequence(allSequences[index][-1], allSequences[index+1][0]))
        
    everything = []
    for sequence1, sequence2 in zip(allSequences, interpolations):
        for line in sequence1:
            everything.append(line)
        for line in sequence2:
            everything.append(line)
    for line in allSequences[-1]:
            everything.append(line)
    # return the angles to euler, and order everything correctly
    # for index in range(0, len(everything)):
    #     rotations, positions = quaternionsAndEulers.separateVectorsSimple(everything[index])
    #     rotationsEuler = quaternionsAndEulers.fromQuaternionToEulerVector(rotations)
    #     everything[index] = quaternionsAndEulers.glueVectors(rotationsEuler, positions, "/home/bee/Desktop/idle animation generator/enekoDataset/genea2023_trn/idle_0000.bvh")
    """
    softEverything = []
    windowSize = 10
    for vecIndex in range(int(windowSize/2+1), int(len(everything)-windowSize/2)):
        averageVector = np.zeros(len(everything[0]))
        for vec in range(int(-windowSize/2), int(+windowSize/2)):
            averageVector = averageVector + everything[vecIndex + vec]
        averageVector = averageVector / windowSize
        softEverything.append(averageVector.copy())
    with open("resultBvhs/softRandomGenerator.bvh", "w") as f:
        f.write(header)
        for line in softEverything:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close
    """
    with open("resultBvhs/randomGenerator.bvh", "w") as f:
        f.write(header)
        for sequence in everything:
            f.write(str(sequence.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close
main()