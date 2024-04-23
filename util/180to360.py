import numpy as np
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import os

def convertFile180to360(path, writePath):
    header, data, _ = bvhLoader.loadBvhToList(path = path, returnData=True, returnHeader=True)
    for row in data:
        for num in range(0, len(row)):
            if not (num%6==0 or (num-1)%6==0 or (num-2)%6==0): # if the number is an angle (indexes 3,4,5)
                if(row[num]<0): # if the number is negative
                    row[num] = 360 + row[num] # make number positive

    # create the directories if they don't exist
    os.makedirs(os.path.dirname(writePath), exist_ok=True)
       
    # write the new bvh
    with open(writePath, "w") as f:
        f.write(header)
        for row in data:
            f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")


if __name__ == "__main__":

    # train split processing old filename
    nameBvhInterlocutor = "silenceDataset3sec/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"
    writeDirPathInterlocutor = "silenceDataset3sec360/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"

    nameBvhMainAgent = "silenceDataset3sec/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"
    writeDirPathMainAgent = "silenceDataset3sec360/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"

    # interlocutor
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        newBvhFilename = os.path.join(writeDirPathInterlocutor, filename)

        if os.path.isfile(bvhFilename):
            convertFile180to360(bvhFilename, newBvhFilename)

    # main agent
    for filename in sorted(os.listdir(nameBvhMainAgent)):
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        newBvhFilename = os.path.join(writeDirPathMainAgent, filename)

        if os.path.isfile(bvhFilename):
            convertFile180to360(bvhFilename, newBvhFilename)

    # val split processing old filename
    nameBvhInterlocutor = "silenceDataset3sec/genea2023_val/genea2023_dataset/val/interloctr/bvh/"
    writeDirPathInterlocutor = "silenceDataset3sec360/genea2023_val/genea2023_dataset/val/interloctr/bvh/"

    nameBvhMainAgent = "silenceDataset3sec/genea2023_val/genea2023_dataset/val/main-agent/bvh/"
    writeDirPathMainAgent = "silenceDataset3sec360/genea2023_val/genea2023_dataset/val/main-agent/bvh/"

    # interlocutor
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        newBvhFilename = os.path.join(writeDirPathInterlocutor, filename)

        if os.path.isfile(bvhFilename):
            convertFile180to360(bvhFilename, newBvhFilename)

    # main agent
    for filename in sorted(os.listdir(nameBvhMainAgent)):
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        newBvhFilename = os.path.join(writeDirPathMainAgent, filename)

        if os.path.isfile(bvhFilename):
            convertFile180to360(bvhFilename, newBvhFilename)