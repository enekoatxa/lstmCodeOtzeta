import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import quaternionsAndEulers
from numpy.random import randint
import numpy as np
import math
from scipy.spatial.transform import Rotation as R 
from scipy.spatial.transform import Slerp
import copy
import subprocess
import re
import random
class RNG():
    def __init__(self):
        self.n_seq = 100
        self.seq_length = 200
        self.interpolation_length = 100
        self.interpolation_length_for_actions = 50
        self.animation_length = 3000
        self.num_animations = 3
        self.loop = False
        self.action_list = []
        self.action_indexes = []
        self.baseSequence = []
        self.finalSequence = []
        self.softSequence = []
       
        # load the idle dataset
        self.namesIdles, self.headersIdles, self.idles, self.masksIdles, self.idsIdles = bvhLoader.loadDatasetForBlendingApp(datasetName="enekoDatasetNoHandsCen", partition="Idle",
                                                    verbose=True, onlyPositions=False, onlyRotations=True, returnHeader = False,
                                                    returnData = True, returnMask = True, returnCounter = False,
                                                    jump = 0, useQuaternions = False, reorderVectors = False)
       # load the idle2 dataset
        self.namesIdles2, self.headersIdles2, self.idles2, self.masksIdles2, self.idsIdles2 = bvhLoader.loadDatasetForBlendingApp(datasetName="enekoDatasetNoHandsCen", partition="Idle2",
                                                    verbose=True, onlyPositions=False, onlyRotations=True, returnHeader = False,
                                                    returnData = True, returnMask = True, returnCounter = False,
                                                    jump = 0, useQuaternions = False, reorderVectors = False)
        # load the actions
        self.namesActions, self.headersActions, self.actions, self.masksActions, self.idsActions = bvhLoader.loadDatasetForBlendingApp(datasetName="enekoDatasetNoHandsCen", partition="Actions",
                                                    verbose=True, onlyPositions=False, onlyRotations=True, returnHeader = False,
                                                    returnData = True, returnMask = True, returnCounter = False,
                                                    jump = 0, useQuaternions = False, reorderVectors = False)
        # load the header of a BVH, to write later automatically
        self.header = bvhLoader.loadBvhToList("/home/bee/Desktop/enekoDatasetHeaderBvhView.bvh", returnHeader = True, returnData = False, returnCounter = False, 
                    onlyPositions = False, onlyRotations = True, jump = 0)
        
    # TODO: interpolation length aldatu
    def setState(self, animation_length, animation_piece_length, interpolation_length, interpolation_length_for_actions, num_animations, action_list, loop):
        self.seq_length = animation_piece_length
        self.n_seq = int(animation_length / (animation_piece_length - interpolation_length)) + (animation_length % (animation_piece_length - interpolation_length) > 0)
        self.animation_length = animation_length
        self.interpolation_length = interpolation_length
        self.interpolation_length_for_actions = interpolation_length_for_actions
        self.num_animations = num_animations
        self.action_list = action_list
        # prepare the index list from the text received
        self.action_indexes.clear()
        for i in range(0, len(self.namesActions)):
            if(any(act in self.namesActions[i] for act in self.action_list)):
                self.action_indexes.append(i)
        self.loop = loop
    
    # EASING FUNCTIONS
    def easingFunction(self, x, easingType = "inOutQuint"):
        match easingType:
            case "inOutCubic":
                return self.easeInOutCubic(x)
            case "inBack":
                return self.easeInBack(x)
            case "inOutBack":
                return self.easeInOutBack(x)
            case "inOutQuint":
                return self.easeInOutQuint(x)
            case "linear":
                return x
        
    
    def easeInBack(self, x):
        c1 = 1.70158
        c3 = c1 + 1
        ret = c3 * x * x * x - c1 * x * x
        return ret

    def easeInOutBack(self, x):
        c1 = 1.70158
        c2 = c1 * 1.525
        if x < 0.5:
            ret = (math.pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
        else:
            ret = (math.pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2
        return ret

    def easeInOutCubic(self, x):
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - ((-2 * x + 2) ** 3) / 2

    def easeInOutQuint(self, x):
        if x < 0.5:
            return 16 * x * x * x * x * x
        else:
            return 1 - ((-2 * x + 2) ** 5) / 2

    def interpolateSequence(self, vectorA, vectorB):
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
        rotationA, positionA = quaternionsAndEulers.separateVectorsSimple(vectorA, usingQuaternions = False)
        rotationB, positionB = quaternionsAndEulers.separateVectorsSimple(vectorB, usingQuaternions = False)
        # convert to np array
        rotationA = np.array(rotationA)
        rotationB = np.array(rotationB)
        positionA = np.array(positionA)
        positionB = np.array(positionB)
        # convert to quaternions
        rotationA = quaternionsAndEulers.fromEulerToQuaternionVector(rotationA)
        rotationB = quaternionsAndEulers.fromEulerToQuaternionVector(rotationB)
        interpolationSeqSize = self.interpolation_length
        # first, interpolate the positions part
        for step in range(1, interpolationSeqSize+1):
            intermediate_position_vector = positionA + (positionB - positionA) * self.easingFunction(step/interpolationSeqSize)
            interpolationPosition.append(intermediate_position_vector.copy())
            positionA = intermediate_position_vector
        # TODO: then, interpolate the rotations part, using slerp (quaternions need slerp for correct interpolation)
        # to interpolate between the two rotation vectors, we need to do it quaternion by quaternion
        timesForQuaternionInterpolation = [self.easingFunction(step/interpolationSeqSize) for step in range(1, interpolationSeqSize+1)]
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
        
        # lastly, rejoin every vector's rotation and position
        for position, rotation in zip(interpolationPosition, interpolationRotation):
            # interpolation.append(quaternionsAndEulers.glueVectors(rotationsVector = rotation, positionsVector = position, vectorHeaderPath="/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen/trn/000_idle.bvh"))
            interpolation.append(np.concatenate((rotation, position)))
        return np.asarray(interpolation)
    
    # method that takes two vectors and a percentage, and returns another vector: the spherical interpolation at point "percentage"
    def blendBetweenTwoVectors(self, vectorA, vectorB, percentage):
        interpolation = []
        rotationA, positionA = quaternionsAndEulers.separateVectorsSimple(vectorA, usingQuaternions = False)
        rotationB, positionB = quaternionsAndEulers.separateVectorsSimple(vectorB, usingQuaternions = False)
        # convert to np array
        rotationA = np.array(rotationA)
        rotationB = np.array(rotationB)
        positionA = np.array(positionA)
        positionB = np.array(positionB)
        # convert to quaternions
        rotationA = quaternionsAndEulers.fromEulerToQuaternionVector(rotationA)
        rotationB = quaternionsAndEulers.fromEulerToQuaternionVector(rotationB)

        # calculate the intermediate position
        blendedPosition = positionA + (positionB - positionA) * percentage

        # calculate the intermediate rotations
        blendedRotation = []
        for index in range(0, len(rotationA)-3, 4):
            r = R.from_quat([[rotationA[index], rotationA[index+1], rotationA[index+2], rotationA[index+3]], [rotationB[index], rotationB[index+1], rotationB[index+2], rotationB[index+3]]])        
            slerp = Slerp(times=[0, 1], rotations=r)
            # create the interpolations and convert them to euler angles
            interpolation = slerp(percentage).as_euler("xyz", degrees=True)
            # put the interpolation in a vector again
            blendedRotation.append(interpolation[0])
            blendedRotation.append(interpolation[1])
            blendedRotation.append(interpolation[2])
        return np.concatenate((blendedRotation, blendedPosition))

    # Function that blends two sequences together: the duration of the final sequence will be len(sequenceA) + len(sequenceB) - self.interpolation_length
    def blendSequences(self, sequenceA, sequenceB, specialSequence=False):
        if(not specialSequence): interpolationLength = self.interpolation_length
        else: interpolationLength = self.interpolation_length_for_actions
        # add the first part of the sequenceA (no blending)
        finalSequence = np.asarray(sequenceA[0:len(sequenceA)-interpolationLength])
        # blend the two sequences
        blendedSequence = []
        for i in range(0, interpolationLength):
            blendedSequence.append(self.blendBetweenTwoVectors(vectorA = sequenceA[len(sequenceA)-interpolationLength+i], vectorB = sequenceB[i],
                                                        percentage = self.easingFunction((i+1)/interpolationLength, "inOutCubic")))
        # blendedSequence = [(x*(index/interpolationLength))+(y*(1-(index/interpolationLength))) for index, (x,y) in enumerate(zip(sequenceA[len(sequenceA)-interpolationLength:], sequenceB[0:interpolationLength]))]
        # blendedSequence = np.append(sequenceA[len(sequenceA)-interpolationLength:], sequenceB[0:interpolationLength], axis=0)
        finalSequence = np.append(finalSequence, blendedSequence, axis=0) 
        # add the last part of the sequenceB (no blending)
        finalSequence = np.append(finalSequence, sequenceB[interpolationLength:], axis=0)
        return np.asarray(finalSequence)

    # Takes the base sequence and introduces a special action sequence and interpolates it
    def introduceSpecialSequence(self, baseSequence, specialSequence, startFrame):
        returnSequence = self.blendSequences(np.asarray(baseSequence[0:startFrame]), specialSequence, specialSequence=True)
        returnSequence = self.blendSequences(returnSequence, baseSequence[startFrame:], specialSequence=True)
        return returnSequence
    
    def generateBaseSequence(self):
        # select random sequences
        allSequences = []
        index = 0
        while(index<self.n_seq):
            personId = randint(0, len(self.idles))
            seqId = randint(0, len(self.idles[personId])-self.seq_length)
            if(self.masksIdles[personId][seqId]==1 and self.masksIdles[personId][seqId+self.seq_length]==1):
                print("selected animation: " + str(personId) + "_" + str(seqId))
                allSequences.append(self.idles[personId][seqId:seqId+self.seq_length])
                index+=1
            else:
                print("NON SELECTED ANIMATION: " + str(personId) + "_" + str(seqId))
        '''
        interpolations = []
        newInterpolation = []
        # create interpolators
        for index in range(0, len(allSequences)-1):
            newInterpolation = self.interpolateSequence(allSequences[index][-self.interpolation_length], allSequences[index+1][self.interpolation_length])
            interpolations.append(copy.deepcopy(newInterpolation))    
        
        everything = np.array(allSequences[0])

        for sequenceIndex in range (0, len(allSequences)-1):
            # blend the interpolation to everything
            everything = self.blendSequences(sequenceA=everything, sequenceB=interpolations[sequenceIndex])
            # blend the next sequence to everything
            everything = self.blendSequences(sequenceA=everything, sequenceB=allSequences[sequenceIndex+1])
        '''

        everything = np.array(allSequences[0])
        for sequenceIndex in range (0, len(allSequences)-1):
            # blend the interpolation to everything
            everything = self.blendSequences(sequenceA=everything, sequenceB=allSequences[sequenceIndex+1])

        # cut the final part of the animation
        everything = everything[0:self.animation_length]

        # if we want the animation to loop, add one final blending sequence
        if(self.loop):
            everything = self.blendSequences(sequenceA=everything, sequenceB=allSequences[0][0:self.interpolation_length+1])
            everything = everything[self.interpolation_length:]
        self.baseSequence = everything
        self.finalSequence = everything
        
    def addSpecialAnimation(self, entryPoint):
        # TODO: introduce a special sequence
        # first, choose a random sequence from the chosen animation types
        animationIndex = random.choice(self.action_indexes)
        actionToIntroduce = self.actions[animationIndex]
        self.finalSequence = self.introduceSpecialSequence(self.finalSequence, actionToIntroduce, entryPoint)

    def deleteAllSpecialSequences(self):
        self.finalSequence = self.baseSequence

    def softenSequence(self):
        softEverything = []
        windowSize = 20
        for vecIndex in range(int(windowSize/2+1), int(len(self.finalSequence)-windowSize/2)):
            averageVector = np.zeros(len(self.finalSequence[0]))
            for vec in range(int(-windowSize/2), int(+windowSize/2)):
                averageVector = averageVector + self.finalSequence[vecIndex + vec]
            averageVector = averageVector / windowSize
            softEverything.append(averageVector.copy())
        self.softSequence = softEverything

    def reorderBaseSequence(self):
        reorderedEverything = []
        for sequenceIndex in range(0, len(self.baseSequence)):
            rotVec, posVec = quaternionsAndEulers.separateVectorsSimple(self.baseSequence[sequenceIndex], usingQuaternions = False)
            reorderedEverything.append(copy.deepcopy(posVec) + copy.deepcopy(rotVec))
        self.baseSequence = reorderedEverything

    def reorderFinalSequence(self):
        reorderedEverything = []
        for sequenceIndex in range(0, len(self.finalSequence)):
            rotVec, posVec = quaternionsAndEulers.separateVectorsSimple(self.finalSequence[sequenceIndex], usingQuaternions = False)
            reorderedEverything.append(copy.deepcopy(posVec) + copy.deepcopy(rotVec))
        self.finalSequence = reorderedEverything

    def reorderSoftSequence(self):
        reorderedEverything = []
        for sequenceIndex in range(0, len(self.softSequence)):
            rotVec, posVec = quaternionsAndEulers.separateVectorsSimple(self.softSequence[sequenceIndex], usingQuaternions = False)
            reorderedEverything.append(copy.deepcopy(posVec) + copy.deepcopy(rotVec))
        self.softSequence = reorderedEverything

    def writeSequence(self, animationIndex):
        self.reorderFinalSequence()
        self.header = re.sub("Frames\:.*\n", f"Frames: {len(self.finalSequence)}\n", self.header)
        path = f"resultBvhs/randomGenerator_{animationIndex}.bvh"
        with open(path, "w") as f:
            f.write(self.header)
            for vector in self.finalSequence:
                f.write(str(["%.6f" % member for member in vector]).replace("[", "").replace("]", "").replace(",", "").replace("'", ""))
                f.write("\n")
            f.close

    def writeSoftSequence(self, animationIndex):
        self.softenSequence()
        self.header = re.sub("Frames\:.*\n", f"Frames: {len(self.softSequence)}\n", self.header)
        path = f"resultBvhs/softRandomGenerator_{animationIndex}.bvh"
        with open(path, "w") as f:
            f.write(self.header)
            for line in self.softSequence:
                f.write(str(["%.6f" % member for member in line]).replace("[", "").replace("]", "").replace(",", "").replace("'", ""))
                f.write("\n")
            f.close

    def openInBlender(self):
        subprocess.run("blender --python blenderImporter.py", shell=True)