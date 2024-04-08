import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import quaternionsAndEulers
#############################################################################################
# global variables: indexes of hands and face, to select what parts of the skeleton to load #
#############################################################################################
faceIndexes = list(range(8, 27)) # face: from 8 to 26
rightHandIndexes = list(range(34, 50)) # right hand: from 34 to 49
leftHandIndexes = list(range(58, 74)) # left hand: from 58 to 73
# precomputed indexes
faceAndHandsIndexesNumbersGenea = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443]
###############################################
# part 1: reading a single bvh file to a list #
###############################################

# calculates which indexes are angles and which are positions from a header 
def calculatePositionIndexesFromHeader(path):
    currentIndex = 0
    positionIndexes = []
    rotationIndexes = []
    with open(path, "r") as f:
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

# reads a bvh file and separates the data from the header. Returns the header if needed, else returns only the data in a list. Can also return just the header.
# each row of the list contains a number of joint rotations. It also returns the number of frames loaded
def loadBvhToList(path, returnHeader = False, returnData = True, returnCounter = True, 
                  onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0, useQuaternions=False):
    ### HEADER ###
    # read and save the header in a variable
    with open(path, "r") as f:
        header = ""
        line = f.readline()
        # read the header until the line "Frame Time: 0.0333333"
        while line.split(" ")[0]!= "Frame":
            header += line
            line = f.readline()
        # add the last header line manually
        # header += line.split("\n")[0]
        header += line

        ### DATA ###
        # read all the rotation data to a list
        data = []
        line = f.readline().replace("\n", "")
        counter = 0
        while True:
            data.append(line.split(" ")[:-1])
            line = f.readline().replace("\n", "")
            counter+=1
            # jumping lines
            if(jump>0):
                for j in range(0, jump):
                    line = f.readline()
                    if not line:break
            if not line: break
        data = [[np.float32(s) for s in sublist] for sublist in data]

    # separate the data into rotations and positions
    rotationData = []
    positionData = []
    positionIndexes, rotationIndexes = calculatePositionIndexesFromHeader(path=path)
    for vector in data:
        rotation, position = quaternionsAndEulers.separateVector(vector, path)
        rotationData.append(rotation.copy())
        positionData.append(position.copy())

    # convert the angles vector to quaternions if wanted
    if(useQuaternions):
        for index in range(0, len(rotationData)):
            rotationData[index] = quaternionsAndEulers.fromEulerToQuaternionVector(rotationData[index])
        # glue the vectors again together, in the most simple way
        for index in range(0, len(data)):
            data[index] = quaternionsAndEulers.concatenateVectorsSimple(rotationsVector=rotationData[index], positionsVector=positionData[index])

    # remove hands and face if needed
    if removeHandsAndFace:
        # THIS IS COMMENTED because the indexes have been precomputed and set as a global variable
        # # with the hands and face indexes, now select ALL the numbers (3 positions and 3 rotations) corresponding to them
        # # concatenate all joint indexes
        # faceAndHandsIndexes = faceIndexes + rightHandIndexes + leftHandIndexes
        # # convert them to 6 numbers each
        # faceAndHandsIndexesNumbers = []
        # for index in faceAndHandsIndexes:
        #     initialIndex = index * 6
        #     faceAndHandsIndexesNumbers.append(initialIndex)
        #     faceAndHandsIndexesNumbers.append(initialIndex+1)
        #     faceAndHandsIndexesNumbers.append(initialIndex+2)
        #     faceAndHandsIndexesNumbers.append(initialIndex+3)
        #     faceAndHandsIndexesNumbers.append(initialIndex+4)
        #     faceAndHandsIndexesNumbers.append(initialIndex+5)
        for vectorIndex in range(0, len(data)):
            data[vectorIndex] = np.delete(data[vectorIndex], faceAndHandsIndexesNumbersGenea, axis=0).copy()

    if onlyPositions:
        data = positionData
            
    if onlyRotations:
        data = rotationData
    
    data = np.asanyarray(data)
    if returnHeader and returnData:
        if returnCounter:
            return header, data, counter
        return header, data
    if returnData:
        if returnCounter:
            return data, counter
        return data
    if returnCounter:
        return header, counter
    return header

#####################################################################
# part 2: loading a dataset partition, in bulk or divided by person #
#####################################################################

# loads the specified bvh dataset, and the partition can also be specified (if trim = true, all sequences are trimmed to the length of the smallest sequence)
# returns: data format is a list of 3 dimensions [n(number of bvh) person][m (depends on the bvh) frame][498 rotation]
def loadDataset(datasetName, partition = "All", specificSize=-1, verbose = False, trim = False, specificTrim = -1, 
                onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, scaler = None, 
                loadDifferences = False, jump = 0, useQuaternions = False):
    allData = []
    allIds = []
    idPerson = 0
    finalTrimSize = 999999999999999
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_val"
    for root, dirs, files in os.walk(path):
        for filename in files:
            if specificSize!=-1 and idPerson>=specificSize:
                    break
            if verbose:
                print(f"Loading file: {filename}")
            if(os.path.splitext(filename).lower()==".bvh"):

                bvhData, bvhSize = loadBvhToList(os.path.join(root, filename), onlyPositions=onlyPositions, onlyRotations=onlyRotations, removeHandsAndFace=removeHandsAndFace, jump=jump, useQuaternions=useQuaternions)
                # if the trim flag is on, calculate the size of the smallest bvh
                if trim:
                    if finalTrimSize > bvhSize:
                        finalTrimSize = bvhSize
                allData.append(bvhData)
                allIds.append(idPerson) # TODO: IDs should not be numbers. Change to one-hot encoding or other
                    # return allData, np.asarray(allIds)
                idPerson+=1

    # after loading the entire dataset, if trim is activated, trim all sequences to the smallest size
    if trim:
        if specificTrim > -1 and specificTrim<=finalTrimSize:
            if verbose:
                print(f"Trimming to size: {specificTrim}")
            # trim
            for person in range(0, len(allData)):
                allData[person] = allData[person][0:specificTrim].copy()
        else:
            if verbose:
                print(f"Trimming to size: {finalTrimSize}")
            for person in range(0, len(allData)):
                allData[person] = allData[person][0:finalTrimSize].copy()

    if loadDifferences:
        print("Loading the differences between frames...")
        differencesVector = []
        differencesDataset = []
        firstPersonPositions = []
        for personIndex in range(0, len(allData)):
            for index in range(0, len(allData[personIndex])-1):
                if index == 0:
                    # save the first frame, to add later
                    firstPersonPositions.append(allData[personIndex][0])
                differencesVector.append((allData[personIndex][index]-allData[personIndex][index+1]))
            differencesDataset.append(differencesVector.copy())
            differencesVector.clear()
        if scaler != None:
            for index in range(0, len(differencesDataset)):
                differencesDataset[index] = scaler.transform(differencesDataset[index])
        return differencesDataset, firstPersonPositions, np.asarray(allIds)

    if scaler != None:
        for index in range(0, len(allData)):
            allData[index] = scaler.transform(allData[index])

    firstPersonPositions = []

    return allData, firstPersonPositions, np.asarray(allIds)

# loads the specified bvh dataset, and the partition can also be specified
# the data is loaded in bulk, no difference between different frames or people, just a list of all the vectors
def loadDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace=False, jump=0):
    allData = []
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_val"
    counter = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            if verbose:
                print(f"Loading file: {filename}")
            vectors = loadBvhToList(os.path.join(root, filename), returnCounter=False, removeHandsAndFace=removeHandsAndFace, jump=jump)
            # instead of appending lists representing people, append the vectors individually
            for vector in vectors:
                allData.append(vector)
                if specificSize!=-1 and counter>=specificSize-1:
                    return allData
                counter+=1
    return allData

############################################################
# part 3: specific methods to load different types of data #
############################################################

# creates the second column of the dataset, for each frame, its result (the next frame)
def createResultsFromDataset(dataset):
    newDataset = []
    personResults = []
    for person in dataset:
        for frame in range(0, len(person)-1):
            personResults.append(person[frame+1])
        newDataset.append(personResults.copy())
        personResults.clear()
    return newDataset

# deletes the last row from the dataset, as it does not have an answer (it has no next frame)
def trimLastFrameFromDataset(dataset):
    newDataset = dataset
    for person in newDataset:
        person = np.delete(person, len(person))
    return newDataset

# loads the vectors, and returns the differences between a vector and its next vector (to prepare the scaler)
def loadDifferencesDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False, onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0):
    datasetX, firstPersonPositions, ids = loadDataset(datasetName, partition = partition, specificSize=specificSize, verbose = verbose,
                onlyPositions = onlyPositions, onlyRotations = onlyRotations, removeHandsAndFace = removeHandsAndFace, jump=jump)
    differencesDataset = []
    for person in datasetX:
        for index in range(0, len(person)-1):
            differencesDataset.append(person[index]-person[index+1])
    return differencesDataset

# creates the sequential part of the dataset, and also its result (the next frame)
# for example, if seq_size = 10, for each frame, it will take 10 frames starting on the initial frame,
# it will create a list with those 10 sequential frames, and the 11th frame will be returned in the result list
def createSequenceFromFataset(dataset, ids, sequenceSize = 10, outSequenceSize = 1):
    sequencedDataset = []
    sequencedDatasetResults = []
    sequencedIds = []
    for person, id in zip(dataset, ids):
        for frame in range(0, len(person)):
            end_ix = frame + sequenceSize
            out_end_ix = end_ix + outSequenceSize
            if out_end_ix > len(person)-1:
                break
            seq_x, seq_y = person[frame:end_ix], person[end_ix:out_end_ix]
            sequencedDataset.append(seq_x.copy())
            sequencedIds.append(id)
            sequencedDatasetResults.append(seq_y.copy())
    return sequencedDataset, sequencedDatasetResults, sequencedIds

def createAndFitStandardScaler(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace = False):
    print("Creating standard scaler")
    scaler = StandardScaler(copy=False)
    print("Loading bulk dataset")
    datasetX = loadDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, removeHandsAndFace=removeHandsAndFace)
    print("Fitting scaler")
    scaler = scaler.fit(datasetX)
    print("Fitted")
    return scaler

def createAndFitStandardScalerForDifferences(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace = False, jump=0, onlyPositions=False, onlyRotations=False):
    print("Creating standard scaler")
    scaler = StandardScaler(copy=False)
    print("Loading bulk dataset with differences")
    datasetX = loadDifferencesDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, removeHandsAndFace=removeHandsAndFace, jump=jump, onlyPositions=onlyPositions, onlyRotations=onlyRotations)
    print("Fitting scaler")
    scaler = scaler.fit(datasetX)
    print("Fitted")
    return scaler

# creates a dataset containing sequences of n frames, and the result being the next frame
def loadSequenceDataset(datasetName, partition = "All", specificSize = -1, verbose = False, sequenceSize = 10, trim = False, specificTrim = -1, onlyPositions = False, onlyRotations = False, outSequenceSize=1, removeHandsAndFace = False, scaler = None, loadDifferences = False, jump = 0):
    # load the dataset in one list and the ids in the second one
    dataset, firstPersonFrames, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, trim=trim, specificTrim=specificTrim, onlyPositions=onlyPositions, onlyRotations=onlyRotations, removeHandsAndFace=removeHandsAndFace, loadDifferences=loadDifferences, jump=jump)
    print(f"{len(dataset[0][0])}")
    # create the sequences and results
    datasetX, datasetY, sequencedIds = createSequenceFromFataset(dataset=dataset, ids=ids, sequenceSize=sequenceSize, outSequenceSize=outSequenceSize)
    # normalize the data
    if scaler != None:
        print("Scaling...")
        for index in range(0, len(datasetX)):
            datasetX[index] = scaler.transform(datasetX[index])
        for index in range(0, len(datasetY)):
            datasetY[index] = scaler.transform(datasetY[index])
    return datasetX, datasetY, firstPersonFrames, sequencedIds

# creates a dataset containing sequences of n frames, and the result being the next frame. It also separetes the rotations and positions in two arrays
def loadSequenceDatasetSeparateRotationsAndPositions(datasetName, partition = "All", specificSize = -1, verbose = False, sequenceSize = 10, trim = False, specificTrim = -1, outSequenceSize=1):
    datasetXPositions, datasetYPositions, sequencedIds = loadSequenceDataset(datasetName, partition = partition, specificSize = specificSize, verbose = verbose, sequenceSize = sequenceSize, trim = trim, specificTrim = specificTrim, onlyPositions = True, outSequenceSize=outSequenceSize)
    datasetXRotations, datasetYRotations, sequencedIds = loadSequenceDataset(datasetName, partition = partition, specificSize = specificSize, verbose = verbose, sequenceSize = sequenceSize, trim = trim, specificTrim = specificTrim, onlyRotations = True, outSequenceSize=outSequenceSize)
    return datasetXPositions, datasetXRotations, datasetYPositions, datasetYRotations, sequencedIds

# loads both the dataset, and its result (X, y) and returns them in two separate lists
def loadDatasetAndCreateResults(datasetName, partition = "All", specificSize = -1, verbose = False, trim = False, specificTrim = -1, loadDifferences = False):
    # load the dataset in one list and the ids in the second one
    datasetX, firstPersonPositions, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, trim=trim, specificTrim=specificTrim, loadDifferences=loadDifferences)
    # using that frame list, create the result list
    datasetY = createResultsFromDataset(datasetX)
    # trim the last frame of the original dataset
    datasetX = trimLastFrameFromDataset(datasetX)
    # return the dataset,the results and the ids
    return datasetX, datasetY, firstPersonPositions, ids
'''
# loads only the dataset, with no results and no ids, to train the discrete VAE used as an encoder for GPT
def loadDatasetForVae(datasetName, partition = "All", specificSize = -1, verbose = False):
    datasetX = loadDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose)
    return datasetX'''

if __name__ == "__main__":
    x, y, idPerson = loadSequenceDataset("silenceDataset3sec", partition="Train", specificSize=10, trim=False, sequenceSize=30, verbose=True)
    print(f"{np.shape(x)}")
    print(f"{np.shape(y)}")
    print(f"{np.shape(idPerson)}")
    print(f"{idPerson}")
