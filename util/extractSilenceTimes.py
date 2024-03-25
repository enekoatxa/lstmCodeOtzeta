import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

############################################################################################################################################
# This script reads the genea dataset, and creates a new dataset with the only silent part of the wav files, consisting of just bvh files. #
############################################################################################################################################
def extractSilence(path, threshold = 0.1, minSeconds = 3, plot = True):
    # load the audio
    y, sr = lr.load(path + ".wav", mono=True)
    # center the audio vertically
    mean = np.mean(y)
    y = [value - mean for value in y]
    # create the needed arrays to visually represent data and to return the values
    amplitudes = []
    times = []
    markers = []
    markersTimes = []
    markersFiltered = []
    markersTimesFiltered = []
    markersTimesStartEnd = []
    # minimum time threshold needs to be multiplied by the audio sampling rate
    timeThreshold = sr * minSeconds
    # counter for the minimum time requirements
    counter = 0
    for s in range(0,len(y)):
        # complete arrays
        amplitudes.append(y[s])
        times.append(s)
        if(np.abs(y[s])<threshold):
            # arrays that only contain values within the threshold
            markers.append(y[s])
            markersTimes.append(s)
            counter += 1
        else:
            # from the thresholded values, just load the ones which complete the minimum time requirements
            if(counter>=timeThreshold):
                # load values to visualize
                markersTimesFiltered.append(range(s-counter, s))
                markersFiltered.append(copy.deepcopy(y[s-counter:s]))
                # load the start and end points to return
                markersTimesStartEnd.append([(s-counter)/sr, s/sr])
                # restart the counter
            counter = 0        
    # plot the data
    if(plot):
        # plot of the whole audio signal
        plt.plot(times, amplitudes, 'k')
        # plot each selected part
        for p, q in zip(markersTimesFiltered, markersFiltered):
            plt.plot(p, q)
        plt.show()
    return markersTimesStartEnd

def extractBvhWithTimes(times, path, writeDirPath):
    ### HEADER ###
    # read and save the header in a variable
    f = open(path + ".bvh", "r")
    header = ""
    line = f.readline()
    # read the header until the line "Frame Time: 0.0333333"
    while line.split()[0]!= "Frame":
        header += line
        line = f.readline()
    # add the last header line manually
    # header += line.split("\n")[0]
    header += line

    ### DATA ###
    # read all the rotation data to a list
    data = []
    line = f.readline()
    while True:
        data.append(line)
        line = f.readline()
        if not line: break

    # create the directories if they don't exist
    os.makedirs(os.path.dirname(writeDirPath), exist_ok=True)

    bvhCounter = 0
    for time in times:
        # extract the start and end frames, using the start and end times
        startFrame = round(time[0]/0.0333333)
        endFrame = round(time[1]/0.0333333)
        # write each bvh using the header, and the data between the start and end frames
        newPath = writeDirPath + os.path.basename(path).split('/')[-1] + "_silence_" + str(bvhCounter) + ".bvh"
        bvhCounter+=1
        newFile = open(newPath, "w")
        newFile.write(header)
        for rotationIndex in range(startFrame, endFrame):
            newFile.write(str(data[rotationIndex]))
        newFile.close()
    f.close()

if __name__ == "__main__":

    # train split processing old filename
    nameWavInterlocutor = "dataset/genea2023_trn/genea2023_dataset/trn/interloctr/wav/"
    nameBvhInterlocutor = "dataset/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"
    writeDirPathInterlocutor = "silenceDataset/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"

    nameWavMainAgent = "dataset/genea2023_trn/genea2023_dataset/trn/main-agent/wav/"
    nameBvhMainAgent = "dataset/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"
    writeDirPathMainAgent = "silenceDataset/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"

    # interlocutor
    for filename in sorted(os.listdir(nameWavInterlocutor)):
        wavFilename = os.path.join(nameWavInterlocutor, filename)
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        # checking if it is a file
        print(wavFilename)
        if os.path.isfile(wavFilename):
            times = extractSilence(path = wavFilename.split(".")[0], threshold = 0.05, plot = False)
            extractBvhWithTimes(times, path = bvhFilename.split(".")[0], writeDirPath = writeDirPathInterlocutor)

    # main agent
    for filename in sorted(os.listdir(nameWavMainAgent)):
        #if(int(filename.split("_")[3])>169):
        wavFilename = os.path.join(nameWavMainAgent, filename)
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        # checking if it is a file
        print(wavFilename)
        if os.path.isfile(wavFilename):
            times = extractSilence(path = wavFilename.split(".")[0], threshold = 0.05, plot = False)
            extractBvhWithTimes(times, path = bvhFilename.split(".")[0], writeDirPath = writeDirPathMainAgent)

    # val split processing
    nameWavInterlocutor = "dataset/genea2023_val/genea2023_dataset/val/interloctr/wav/"
    nameBvhInterlocutor = "dataset/genea2023_val/genea2023_dataset/val/interloctr/bvh/"
    writeDirPathInterlocutor = "silenceDataset/genea2023_val/genea2023_dataset/val/interloctr/bvh/"

    nameWavMainAgent = "dataset/genea2023_val/genea2023_dataset/val/main-agent/wav/"
    nameBvhMainAgent = "dataset/genea2023_val/genea2023_dataset/val/main-agent/bvh/"
    writeDirPathMainAgent = "silenceDataset/genea2023_val/genea2023_dataset/val/main-agent/bvh/"

    # interlocutor
    for filename in sorted(os.listdir(nameWavInterlocutor)):
        wavFilename = os.path.join(nameWavInterlocutor, filename)
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        # checking if it is a file
        print(wavFilename)
        if os.path.isfile(wavFilename):
            times = extractSilence(path = wavFilename.split(".")[0], threshold = 0.05, plot = False)
            extractBvhWithTimes(times, path = bvhFilename.split(".")[0], writeDirPath = writeDirPathInterlocutor)

    # main agent
    for filename in sorted(os.listdir(nameWavMainAgent)):
        wavFilename = os.path.join(nameWavMainAgent, filename)
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        # checking if it is a file
        print(wavFilename)
        if os.path.isfile(wavFilename):
            times = extractSilence(path = wavFilename.split(".")[0], threshold = 0.05, plot = False)
            extractBvhWithTimes(times, path = bvhFilename.split(".")[0], writeDirPath = writeDirPathMainAgent)