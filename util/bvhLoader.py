import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import quaternionsAndEulers
#############################################################################################
# global variables: indexes of hands and face, to select what parts of the skeleton to load #
#############################################################################################
faceIndexes = list(range(8, 27)) # face: from 8 to 26
rightHandIndexes = list(range(34, 50)) # right hand: from 34 to 49
leftHandIndexes = list(range(58, 74)) # left hand: from 58 to 73
###############################################
# part 1: reading a single bvh file to a list #
###############################################

# calculates which indexes are angles and which are positions from a header 
def calculatePositionIndexesFromHeader(path):
    currentIndex = 0
    comparisonIndex = 0
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
            comparisonIndex +=1
        line = f.readline()
    return positionIndexes, rotationIndexes

# reads a bvh file and separates the data from the header. Returns the header if needed, else returns only the data in a list. Can also return just the header.
# each row of the list contains a number of joint rotations. It also returns the number of frames loaded
def loadBvhToList(path, returnHeader = False, returnData = True, returnCounter = True, 
                  onlyPositions = False, onlyRotations = False, jump = 0, useQuaternions=False,
                  reorderVectors = True):
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
        # if we only want the header, break the method
        if returnHeader and not returnData and not returnCounter:
            return header
        ### DATA ###
        # read all the rotation data to a list
        data = []
        line = f.readline().rstrip().replace("\n", "")
        counter = 0
        while True:
            data.append(line.rstrip().split(" "))
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
    for vector in data:
        rotation, position = quaternionsAndEulers.separateVector(vector, path)
        rotationData.append(rotation.copy())
        positionData.append(position.copy())

    # setup the quaternionsAndEulers global variables by sticking together two vectors with the correct sizes # VERY VERY IMPORTANT [0:2 is for only rot]
    quaternionsAndEulers.concatenateVectorsSimple(rotationData[0], positionData[0][0:2], usingQuaternions = False)
    quaternionsAndEulers.concatenateVectorsSimple(quaternionsAndEulers.fromEulerToQuaternionVector(rotationData[0]), positionData[0][0:2], usingQuaternions = True)

    # convert the angles vector to quaternions if wanted
    if(useQuaternions):
        for index in range(0, len(rotationData)):
            rotationData[index] = quaternionsAndEulers.fromEulerToQuaternionVector(rotationData[index])
    
    if reorderVectors:
        # glue the vectors again together, in the most simple way
        for index in range(0, len(data)):
            data[index] = quaternionsAndEulers.concatenateVectorsSimple(rotationsVector=rotationData[index], positionsVector=positionData[index], usingQuaternions=useQuaternions)
    
    if onlyPositions:
        data = positionData
            
    if onlyRotations:
        for index in range(0, len(data)):
            data[index] = quaternionsAndEulers.concatenateVectorsSimple(rotationsVector=rotationData[index], positionsVector=positionData[index][0:3], usingQuaternions=useQuaternions)

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
                onlyPositions = False, onlyRotations = False, scaler = None, 
                loadDifferences = False, jump = 0, useQuaternions = False, reorderVectors = True):
    allData = []
    allIds = []
    idPerson = 0
    finalTrimSize = 999999999999999
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/val"
    for root, dirs, files in os.walk(path):
        for filename in files:
            if specificSize!=-1 and idPerson>=specificSize:
                    break
            if verbose:
                print(f"Loading file: {filename}")
            if(os.path.splitext(filename)[1].lower()==".bvh"):
                bvhData, bvhSize = loadBvhToList(os.path.join(root, filename), onlyPositions=onlyPositions, onlyRotations=onlyRotations, jump=jump, useQuaternions=useQuaternions, reorderVectors=reorderVectors)
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

    if loadDifferences: #TODO
        print("Loading the differences between frames...")
        differencesVector = []
        differencesDataset = []
        firstPersonPositions = []
        for personIndex in range(0, len(allData)):
            print(f"person index: {personIndex}")
            for index in range(0, len(allData[personIndex])-1):
                print(f"*****vector index: {index}")
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
def loadDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False, onlyPositions = False, onlyRotations = False, jump = 0, useQuaternions = False, reorderVectors = True):
    allData = []
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/val"
    counter = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            if verbose:
                print(f"Loading file: {filename}")
            vectors = loadBvhToList(os.path.join(root, filename), returnCounter=False, jump=jump, reorderVectors=reorderVectors, onlyPositions=onlyPositions, onlyRotations=onlyRotations, useQuaternions=useQuaternions)
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
def loadDifferencesDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False, onlyPositions = False, onlyRotations = False, jump = 0, useQuaternions = False, reorderVectors = True):
    datasetX, firstPersonPositions, ids = loadDataset(datasetName, partition = partition, specificSize=specificSize, verbose = verbose,
                onlyPositions = onlyPositions, onlyRotations = onlyRotations, jump=jump, useQuaternions=useQuaternions, reorderVectors = reorderVectors)
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

def createAndFitStandardScaler(datasetName, partition = "All", specificSize=-1, verbose = False, jump=0, onlyPositions=False, onlyRotations=False, useQuaternions = False, reorderVectors = True):
    print("Creating standard scaler")
    scaler = StandardScaler()
    print("Loading bulk dataset")
    datasetX = loadDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, reorderVectors=reorderVectors, jump=jump, onlyPositions=onlyPositions, onlyRotations=onlyRotations, useQuaternions=useQuaternions)
    print("Fitting scaler")
    scaler = scaler.fit(datasetX)
    print(f"scaler mean: {scaler.mean_}")
    print("Fitted")
    return scaler

def createAndFitStandardScalerForDifferences(datasetName, partition = "All", specificSize=-1, verbose = False, jump=0, onlyPositions=False, onlyRotations=False, useQuaternions = False, reorderVectors = True):
    print("Creating standard scaler")
    scaler = StandardScaler()
    print("Loading bulk dataset with differences")
    datasetX = loadDifferencesDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, jump=jump, onlyPositions=onlyPositions, onlyRotations=onlyRotations, useQuaternions = useQuaternions, reorderVectors=reorderVectors)
    print(f"{len(datasetX)}")
    print(f"{len(datasetX[0])}")
    print("Fitting scaler")
    scaler = scaler.fit(datasetX)
    print(f"scaler mean: {scaler.mean_}")
    print("Fitted")
    return scaler

# creates a dataset containing sequences of n frames, and the result being the next frame
def loadSequenceDataset(datasetName, partition = "All", specificSize = -1, verbose = False, sequenceSize = 10, trim = False, specificTrim = -1, onlyPositions = False, onlyRotations = False, outSequenceSize=1, scaler = None, loadDifferences = False, jump = 0, useQuaternions = False, reorderVectors = False):
    # load the dataset in one list and the ids in the second one
    dataset, firstPersonFrames, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose,
                                                   trim=trim, specificTrim=specificTrim, onlyPositions=onlyPositions, onlyRotations=onlyRotations,
                                                    loadDifferences=loadDifferences, jump=jump, useQuaternions=useQuaternions, 
                                                    reorderVectors=reorderVectors, scaler=scaler)

    
    # create the sequences and results
    datasetX, datasetY, sequencedIds = createSequenceFromFataset(dataset=dataset, ids=ids, sequenceSize=sequenceSize, outSequenceSize=outSequenceSize)
    # normalize the data
    # if scaler != None:
    #     print("Scaling...")
    #     for index in range(0, len(datasetX)):
    #         datasetX[index] = scaler.transform(datasetX[index])
    #     for index in range(0, len(datasetY)):
    #         datasetY[index] = scaler.transform(datasetY[index])
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
