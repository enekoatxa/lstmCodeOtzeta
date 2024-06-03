import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np

if __name__ == "__main__":

    # train split processing old filename
    nameBvh = "../../enekoDatasetNoHands/idle/"
    writeDirPath = "../../enekoDatasetNoHandsCen/idle/"

    if not os.path.exists(writeDirPath): os.makedirs(writeDirPath)
    
    for filename in sorted(os.listdir(nameBvh)):
        bvhFilename = os.path.join(nameBvh, filename)
        writeBvhFilename = os.path.join(writeDirPath, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            # calculate the Y offset of the skeleton (depends on the size of the skeleton) (positionY variable of bone 0)
            yOffset = float(header.split("\n")[3].split(" ")[2])
            for vector in data:
                # edit the Y position of bone 0
                vector[1] = vector[1] - yOffset
                # edit the rotations of bone 1
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")

    nameBvh = "../../enekoDatasetNoHands/idle2/"
    writeDirPath = "../../enekoDatasetNoHandsCen/idle2/"

    if not os.path.exists(writeDirPath): os.makedirs(writeDirPath)
    
    for filename in sorted(os.listdir(nameBvh)):
        bvhFilename = os.path.join(nameBvh, filename)
        writeBvhFilename = os.path.join(writeDirPath, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            # calculate the Y offset of the skeleton (depends on the size of the skeleton) (positionY variable of bone 0)
            yOffset = float(header.split("\n")[3].split(" ")[2])
            for vector in data:
                # edit the Y position of bone 0
                vector[1] = vector[1] - yOffset
                # edit the rotations of bone 1
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")

    nameBvh = "../../enekoDatasetNoHands/actions/"
    writeDirPath = "../../enekoDatasetNoHandsCen/actions/"

    if not os.path.exists(writeDirPath): os.makedirs(writeDirPath)
    
    for filename in sorted(os.listdir(nameBvh)):
        bvhFilename = os.path.join(nameBvh, filename)
        writeBvhFilename = os.path.join(writeDirPath, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            # calculate the Y offset of the skeleton (depends on the size of the skeleton) (positionY variable of bone 0)
            yOffset = float(header.split("\n")[3].split(" ")[2])
            for vector in data:
                # edit the Y position of bone 0
                vector[1] = vector[1] - yOffset
                # edit the rotations of bone 1
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")

    nameBvh = "../../enekoDatasetNoHands/phone/"
    writeDirPath = "../../enekoDatasetNoHandsCen/phone/"

    if not os.path.exists(writeDirPath): os.makedirs(writeDirPath)
    
    for filename in sorted(os.listdir(nameBvh)):
        bvhFilename = os.path.join(nameBvh, filename)
        writeBvhFilename = os.path.join(writeDirPath, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            # calculate the Y offset of the skeleton (depends on the size of the skeleton) (positionY variable of bone 0)
            yOffset = float(header.split("\n")[3].split(" ")[2])
            for vector in data:
                # edit the Y position of bone 0
                vector[1] = vector[1] - yOffset
                # edit the rotations of bone 1
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")
    
    nameBvh = "../../enekoDatasetNoHands/lookback/"
    writeDirPath = "../../enekoDatasetNoHandsCen/lookback/"

    if not os.path.exists(writeDirPath): os.makedirs(writeDirPath)
    
    for filename in sorted(os.listdir(nameBvh)):
        bvhFilename = os.path.join(nameBvh, filename)
        writeBvhFilename = os.path.join(writeDirPath, filename)
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            # calculate the Y offset of the skeleton (depends on the size of the skeleton) (positionY variable of bone 0)
            yOffset = float(header.split("\n")[3].split(" ")[2])
            for vector in data:
                # edit the Y position of bone 0
                vector[1] = vector[1] - yOffset
                # edit the rotations of bone 1
            with open(writeBvhFilename, "w") as f:
                f.write(header)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")