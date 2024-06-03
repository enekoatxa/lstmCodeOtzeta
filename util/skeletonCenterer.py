import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":

    z180rM = [[-1, -0, 0],[-0, -1, 0],[0, 0, 1]]

    # train split processing old filename
    nameBvhInterlocutor = "../../silenceDataset3secNoHands/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"
    writeDirPathInterlocutor = "../../silenceDataset3secNoHandsCen/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"

    nameBvhMainAgent = "../../silenceDataset3secNoHands/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"
    writeDirPathMainAgent = "../../silenceDataset3secNoHandsCen/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)
    
    # interlocutor (needs the rotations corrected also)
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        writeBvhFilename = os.path.join(writeDirPathInterlocutor, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            firstVector = data[0].copy()
            for vector in data:
                # edit the positions of bone 0
                vector[0] = vector[0] - firstVector[0]
                vector[1] = vector[1] - firstVector[1]
                vector[2] = vector[2] - firstVector[2]
                # change the sign in the x axis movement (necessary for correctly rotating)
                vector[0] = -vector[0]
                # edit the rotations of bone 1
                r = R.from_euler("xyz", [vector[9], vector[10], vector[11]], degrees=True)
                rM = r.as_matrix()
                rMNew = np.matmul(rM, z180rM)
                rNew = R.from_matrix(rMNew)
                vector[9] = rNew.as_euler("xyz", degrees=True)[0]
                vector[10] = rNew.as_euler("xyz", degrees=True)[1]
                vector[11] = rNew.as_euler("xyz", degrees=True)[2]
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")
               
    # main agent
    for filename in sorted(os.listdir(nameBvhMainAgent)):
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        writeBvhFilename = os.path.join(writeDirPathMainAgent, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            firstVector = data [0].copy()
            for vector in data:
                vector[0] = vector[0] - firstVector[0]
                vector[1] = vector[1] - firstVector[1]
                vector[2] = vector[2] - firstVector[2]

            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")


    # validation split processing old filename
    nameBvhInterlocutor = "../../silenceDataset3secNoHands/genea2023_val/genea2023_dataset/val/interloctr/bvh/"
    writeDirPathInterlocutor = "../../silenceDataset3secNoHandsCen/genea2023_val/genea2023_dataset/val/interloctr/bvh/"

    nameBvhMainAgent = "../../silenceDataset3secNoHands/genea2023_val/genea2023_dataset/val/main-agent/bvh/"
    writeDirPathMainAgent = "../../silenceDataset3secNoHandsCen/genea2023_val/genea2023_dataset/val/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)

    # interlocutor (needs the rotations corrected also)
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        writeBvhFilename = os.path.join(writeDirPathInterlocutor, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            firstVector = data [0].copy()
            for vector in data:
                # edit the positions of bone 0
                vector[0] = vector[0] - firstVector[0]
                vector[1] = vector[1] - firstVector[1]
                vector[2] = vector[2] - firstVector[2]
                # change the sign in the x axis movement (necessary for correctly rotating)
                vector[0] = -vector[0]
                # edit the rotations of bone 1
                r = R.from_euler("xyz", [vector[9], vector[10], vector[11]], degrees=True)
                rM = r.as_matrix()
                rMNew = np.matmul(rM, z180rM)
                rNew = R.from_matrix(rMNew)
                vector[9] = rNew.as_euler("xyz", degrees=True)[0]
                vector[10] = rNew.as_euler("xyz", degrees=True)[1]
                vector[11] = rNew.as_euler("xyz", degrees=True)[2]

            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")
            
    # main agent
    for filename in sorted(os.listdir(nameBvhMainAgent)):
        bvhFilename = os.path.join(nameBvhMainAgent, filename)
        writeBvhFilename = os.path.join(writeDirPathMainAgent, filename)
        lineRemovingOn = False
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            firstVector = data [0].copy()
            for vector in data:
                #vector[0] = vector[0] - firstVector[0]
                vector[1] = vector[1] - firstVector[1]
                #vector[2] = vector[2] - firstVector[2]

            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")