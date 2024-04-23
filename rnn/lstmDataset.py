import keras
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np

class lstmDataset(keras.utils.PyDataset):
    def __init__(self, root, datasetName = "silenceDataset3sec", partition = "Validation", isTiny = False, specificSize = -1, trim = False, sequenceSize = -1, verbose = False, specificTrim = -1, batchSize = 56, onlyPositions = False, onlyRotations = False, outSequenceSize=1, scaler = None, loadDifferences = False, jump = 0, useQuaternions = False, **kwargs):
        super().__init__(**kwargs)
        """
        :param datasetName: name of the dataset that we want to load.
        :param partition: name of the partition that we want to load.
        :param is_tiny: for testing purposes
        """

        # load everything
        # x, y, ids = bvhLoader.loadSequenceDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, trim=trim,
        #                                            sequenceSize=sequenceSize, verbose=verbose, specificTrim=specificTrim,
        #                                            onlyPositions=onlyPositions, onlyRotations=onlyRotations, outSequenceSize=outSequenceSize, scaler = scaler)
        
        x, firstPersonFrames, ids = bvhLoader.loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, trim=trim,
                                                verbose=verbose, specificTrim=specificTrim, onlyPositions=onlyPositions,
                                                onlyRotations=onlyRotations, scaler=scaler, loadDifferences = loadDifferences,
                                                jump = jump, useQuaternions = useQuaternions)

        # isTiny
        if isTiny:
            x = x[:10]
            ids = ids[:10]

        # calculate how many frames we have in total, counting all frames from all people
        totalNumberOfFrames = 0
        totalNumberOfSequences = 0
        for person in x:
            totalNumberOfFrames += len(person)
            totalNumberOfSequences += len(person)-sequenceSize # length of the total sequence - sequenceSize + 1 (?) - 1 (for the result)

        self.totalNumberOfSequences = totalNumberOfSequences
        self.totalNumberOfFrames = totalNumberOfFrames
        self.batchSize = batchSize
        self.sequences = x
        #self.firstPersonFrames = firstPersonFrames
        self.ids = ids
        self.scaler = scaler
        self.sequenceSize = sequenceSize
        self.outSequenceSize = outSequenceSize
        self.length = int(totalNumberOfSequences/self.batchSize)
        self.vectorSize = len(x[0][0])
        print(f"vectorsize: {self.vectorSize}")
        with open(1, "w", closefd=False) as f:
            print("dataloader length:" + str(self.length), file=f, flush=True)
            if(self.length<0):
                print("ERROR: YOU HAVE PUT THE SEQUENCE LENGTH BIGGER THAN THE SEQUENCES IN THE DATASET")

    def __getitem__(self, index):
        """
        :param index: index of the person
        :return: tuple(list of all sequences in a bvh file, list of all results for all sequences, id of the person) 
        """
        lowLimit = index * self.batchSize
        highLimit = min(lowLimit + self.batchSize, self.totalNumberOfFrames)

        # instead of returning the frames, we first create the sequences and then return them
        # metodoa oso korapilatsua da
        sequencesToReturn = []
        sequencedDatasetResults = []
        sequencesLastFrame = []
        # first calculate fast from where we need to start: the person index and the frame index
        # the frame index will be the low limit (where we start the loading). Current index will be the same.
        # the person index is where the lowLimit falls. Example: index 1342 falls in person 67
        lastFrameProcessed = lowLimit
        currentIndex = lowLimit
        lastPersonProcessed = 0
        currentSize = 0
        for personIndex in range(0, len(self.sequences)):
            currentSize += len(self.sequences[personIndex])
            if(currentSize<=lowLimit):
                lastFrameProcessed -= len(self.sequences[personIndex])
                lastPersonProcessed += 1

        for person in range(lastPersonProcessed, len(self.sequences)):
            for frame in range(lastFrameProcessed, len(self.sequences[person])):
                if(currentIndex>=lowLimit):
                    # if the index has surpassed the highLimit, break the loop completely and return
                    if(currentIndex>=highLimit):
                        break
                    # else, calculate the sequence and add it to the returning array
                    end_ix = frame + self.sequenceSize
                    out_end_ix = end_ix + self.outSequenceSize
                    # if in this person, we cant calculate the sequence, break the loop, and continue calculating the next one, start from frame 0 in the next sequence
                    if end_ix > len(self.sequences[person])-1 or out_end_ix > len(self.sequences[person])-1:
                        lastFrameProcessed = 0
                        break
                    # print("processing: " + str(frame) + ":: limits :: " + str(end_ix) + "|" + str(out_end_ix) + "||person: " + str(person))
                    seq_x, seq_x_last, seq_y = self.sequences[person][frame:end_ix], self.sequences[person][end_ix-1:end_ix], self.sequences[person][end_ix:out_end_ix]
                    #reshape seq_y
                    # seq_y = np.squeeze(seq_y, axis=0) # TODO hau deskomentatu inferentzia normalerako
                    sequencesToReturn.append(seq_x.copy())
                    sequencesLastFrame.append(seq_x_last.copy())
                    sequencedDatasetResults.append(seq_y.copy())
                currentIndex += 1

            # if the index has surpassed the highLimit, break the loop completely and return
            if(currentIndex>=highLimit):
                break
        # after calculating the sequences and the results (they don't load in memory), we return them
        # print(np.array(sequencesToReturn).shape)
        # print(np.array(sequencedDatasetResults).shape)

        # sometimes the method fails, i don't know why. If the return size is 0, I will return a blank example, not to break the method
        if(len(np.array(sequencesToReturn)) == 0):
            sequencesToReturn = np.zeros((self.batchSize, self.sequenceSize, self.vectorSize))
            sequencesLastFrame = np.zeros((self.batchSize, self.sequenceSize, self.vectorSize))            
            sequencedDatasetResults = np.zeros((self.batchSize, self.outSequenceSize, self.vectorSize))
            print("an example has been broken")

        return (np.array(sequencesToReturn), np.array(sequencesLastFrame)), np.array(sequencedDatasetResults)

    def __len__(self):
        return self.length
    
    def getScaler(self):
        return self.scaler
