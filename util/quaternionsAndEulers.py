from scipy.spatial.transform import Rotation as R 
import numpy as np
import timeit

##################################################################
# global variables for glueing and separating quaternion vectors #
##################################################################
numberOfRotationValuesQuaternions = 0
numberOfRotationValuesEulers = 0
numberOfPositionValues = 0
# faceAndHandsIndexesNumbersGenea = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443]

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

def separateVector(wholeVector, vectorHeaderPath="null"):
    if vectorHeaderPath == "null":
        raise NameError(f"The vectorHeaderPath to read the angles and rotations indexes has not been set.")
    positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader(vectorHeaderPath)
    rotationsVector = []
    positionsVector = []
    for index in range(0, len(wholeVector)):
        if(index in rotationIndexes):
            rotationsVector.append(wholeVector[index])
        if(index in positionIndexes):
            positionsVector.append(wholeVector[index])
    return rotationsVector, positionsVector

def glueVectors(rotationsVector, positionsVector, vectorHeaderPath="null"):
    if vectorHeaderPath == "null":
        raise NameError(f"The vectorHeaderPath to read the angles and rotatoins indexes has not been set.")
    positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader(vectorHeaderPath)
    wholeVector = [None] * (len(positionIndexes) + len(rotationIndexes))
    for index in range(0, len(rotationsVector)):
        wholeVector[rotationIndexes[index]] = rotationsVector[index]
    for index in range(0, len(positionsVector)):
        wholeVector[positionIndexes[index]] = positionsVector[index]
    return wholeVector

# this is faster than list comprehensions (tested with timeit)
def concatenateVectorsSimple(rotationsVector, positionsVector, usingQuaternions = False):
    wholeVector = []
    for rotation in rotationsVector:
        wholeVector.append(rotation)
    for position in positionsVector:
        wholeVector.append(position)
    # change the global variables, so we know where to separate later
    if(usingQuaternions):
        global numberOfRotationValuesQuaternions
        numberOfRotationValuesQuaternions = len(rotationsVector)
        # print(f"setting quat rotation number to: {numberOfRotationValuesQuaternions}")
    else:
        global numberOfRotationValuesEulers
        numberOfRotationValuesEulers = len(rotationsVector)
        # print(f"setting euler rotation number to: {numberOfRotationValuesEulers}")
    global numberOfPositionValues
    numberOfPositionValues= len(positionsVector)
    # print(f"setting position number to: {numberOfPositionValues}")
    return wholeVector

def separateVectorsSimple(wholeSimpleVector, usingQuaternions = False):
    rotationsVector = []
    positionsVector = []
    if usingQuaternions:
        for index in range(0, len(wholeSimpleVector)):
            if(index<numberOfRotationValuesQuaternions):
                rotationsVector.append(wholeSimpleVector[index])
            else:
                positionsVector.append(wholeSimpleVector[index])
    else:
        for index in range(0, len(wholeSimpleVector)):
            if(index<numberOfRotationValuesEulers):
                rotationsVector.append(wholeSimpleVector[index])
            else:
                positionsVector.append(wholeSimpleVector[index])
    return rotationsVector, positionsVector

def separateVectorsSimpleForLoss(wholeSimpleVector, usingQuaternions = False):
    rotationsVector = []
    positionsVector = []
    if usingQuaternions:
        for index in range(0, wholeSimpleVector.shape[1]):
            if(index<numberOfRotationValuesQuaternions):
                rotationsVector.append(wholeSimpleVector[index])
            else:
                positionsVector.append(wholeSimpleVector[index])
    else:
        for index in range(0, wholeSimpleVector.shape[1]):
            if(index<numberOfRotationValuesEulers):
                rotationsVector.append(wholeSimpleVector[index])
            else:
                positionsVector.append(wholeSimpleVector[index])
    return rotationsVector, positionsVector