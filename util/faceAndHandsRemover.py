import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np
# values to remove hands and face for the genea challenge database
faceAndHandsIndexesNumbersGenea = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443]
faceIndexes = list(range(8, 27)) # face: from 8 to 26
rightHandIndexes = list(range(34, 50)) # right hand: from 34 to 49
leftHandIndexes = list(range(58, 74)) # left hand: from 58 to 73

faceIndex = 8
rightHandIndex = 34
leftHandIndex = 58

if __name__ == "__main__":

    # train split processing old filename
    nameBvhInterlocutor = "../../silenceDataset3sec/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"
    writeDirPathInterlocutor = "../../silenceDataset3secNoHands/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/"

    nameBvhMainAgent = "../../silenceDataset3sec/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"
    writeDirPathMainAgent = "../../silenceDataset3secNoHands/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)
    
    # interlocutor
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        writeBvhFilename = os.path.join(writeDirPathInterlocutor, filename)
        lineRemovingOn = False
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            newHeader = ""
            # reformat the header (remove the unwanted bones)
            jointCounter = 0
            parenthesesCounter = 0
            for line in header.split("\n"):
                if "JOINT" in line.split(" "):
                    jointCounter += 1
                if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
                    if "JOINT" in line.split(" "):
                        parenthesesCounter +=1
                    lineRemovingOn = True
                if lineRemovingOn:
                    if "{" in line.split(" "):
                        parenthesesCounter +=1
                    if "}" in line.split(" "):
                        parenthesesCounter -=1
                    if parenthesesCounter == 0:
                        lineRemovingOn = False
                if not lineRemovingOn: 
                    newHeader += line
                    newHeader += "\n"
            # remove the last \n from the new header
            newHeader = newHeader[:-1]
            # reformat the data (remove the unused numbers)
            data = data.tolist()
            for index in range(0, len(data)):
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
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
            newHeader = ""
            # reformat the header (remove the unwanted bones)
            jointCounter = 0
            parenthesesCounter = 0
            for line in header.split("\n"):
                if "JOINT" in line.split(" "):
                    jointCounter += 1
                if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
                    if "JOINT" in line.split(" "):
                        parenthesesCounter +=1
                    lineRemovingOn = True
                if lineRemovingOn:
                    if "{" in line.split(" "):
                        parenthesesCounter +=1
                    if "}" in line.split(" "):
                        parenthesesCounter -=1
                    if parenthesesCounter == 0:
                        lineRemovingOn = False
                if not lineRemovingOn: 
                    newHeader += line
                    newHeader += "\n"
            # remove the last \n from the new header
            newHeader = newHeader[:-1]
            # reformat the data (remove the unused numbers)
            data = data.tolist()
            for index in range(0, len(data)):
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")


    # validation split processing old filename
    nameBvhInterlocutor = "../../silenceDataset3sec/genea2023_val/genea2023_dataset/val/interloctr/bvh/"
    writeDirPathInterlocutor = "../../silenceDataset3secNoHands/genea2023_val/genea2023_dataset/val/interloctr/bvh/"

    nameBvhMainAgent = "../../silenceDataset3sec/genea2023_val/genea2023_dataset/val/main-agent/bvh/"
    writeDirPathMainAgent = "../../silenceDataset3secNoHands/genea2023_val/genea2023_dataset/val/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)

    # interlocutor
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        writeBvhFilename = os.path.join(writeDirPathInterlocutor, filename)
        lineRemovingOn = False
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            newHeader = ""
            # reformat the header (remove the unwanted bones)
            jointCounter = 0
            parenthesesCounter = 0
            for line in header.split("\n"):
                if "JOINT" in line.split(" "):
                    jointCounter += 1
                if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
                    if "JOINT" in line.split(" "):
                        parenthesesCounter +=1
                    lineRemovingOn = True
                if lineRemovingOn:
                    if "{" in line.split(" "):
                        parenthesesCounter +=1
                    if "}" in line.split(" "):
                        parenthesesCounter -=1
                    if parenthesesCounter == 0:
                        lineRemovingOn = False
                if not lineRemovingOn: 
                    newHeader += line
                    newHeader += "\n"
            # remove the last \n from the new header
            newHeader = newHeader[:-1]
            # reformat the data (remove the unused numbers)
            data = data.tolist()
            for index in range(0, len(data)):
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
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
            newHeader = ""
            # reformat the header (remove the unwanted bones)
            jointCounter = 0
            parenthesesCounter = 0
            for line in header.split("\n"):
                if "JOINT" in line.split(" "):
                    jointCounter += 1
                if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
                    if "JOINT" in line.split(" "):
                        parenthesesCounter +=1
                    lineRemovingOn = True
                if lineRemovingOn:
                    if "{" in line.split(" "):
                        parenthesesCounter +=1
                    if "}" in line.split(" "):
                        parenthesesCounter -=1
                    if parenthesesCounter == 0:
                        lineRemovingOn = False
                if not lineRemovingOn: 
                    newHeader += line
                    newHeader += "\n"
            # remove the last \n from the new header
            newHeader = newHeader[:-1]
            # reformat the data (remove the unused numbers)
            data = data.tolist()
            for index in range(0, len(data)):
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")