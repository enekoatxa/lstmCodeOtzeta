import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np
# values to remove hands and face for the genea challenge database
faceAndHandsIndexesNumbersGenea = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443]
# values to remove hands and face for the recorded database
faceAndHandsIndexesNumbersEneko = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567]
faceIndexes = list(range(5, 96)) # face: from 8 to 26
rightHandIndexes = list(range(123, 144)) # right hand: from 34 to 49
leftHandIndexes = list(range(99, 120)) # left hand: from 58 to 73

faceIndex = 5
rightHandIndex = 123
leftHandIndex = 99

def calculateIndexes():
    allIndexes = []
    deletingOn = False
    counterJoint = 0
    counterIndex = 0
    bvhFilename = "../../enekoDataset/genea2023_trn/000_idle.bvh"
    header = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=False, returnCounter=False, reorderVectors=False)
    for line in header.split("\n"):
        if("JOINT" in line):
            counterJoint += 1

        if(counterJoint in faceIndexes or counterJoint in rightHandIndexes or counterJoint in leftHandIndexes):
            deletingOn = True
        else:
            deletingOn = False

        if("position" in line or "rotation" in line):
            for word in line.split(" "):
                if "position" in word or "rotation" in word:
                    counterIndex +=1
                if deletingOn:
                    allIndexes.append(counterIndex)
    allIndexes = list(dict.fromkeys(allIndexes))
    print(allIndexes)
             
if __name__ == "__main__":

    # train split processing old filename
    nameBvhInterlocutor = "../../enekoDataset/genea2023_trn/"
    writeDirPathInterlocutor = "../../enekoDatasetNoHands/genea2023_trn/"

    # nameBvhMainAgent = "../../silenceDataset3sec/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"
    # writeDirPathMainAgent = "../../silenceDataset3secNoHands/genea2023_trn/genea2023_dataset/trn/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    # if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)
    
    # interlocutor
    for filename in sorted(os.listdir(nameBvhInterlocutor)):
        bvhFilename = os.path.join(nameBvhInterlocutor, filename)
        writeBvhFilename = os.path.join(writeDirPathInterlocutor, filename)
        lineRemovingOn = False
        # checking if it is a file
        print(bvhFilename)
        if os.path.isfile(bvhFilename):
            header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
            header = header.replace("\t", "")
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
                        print(parenthesesCounter)
                    lineRemovingOn = True
                if lineRemovingOn:
                    if "{" in line.split(" "):
                        parenthesesCounter +=1
                    if "}" in line.split(" "):
                        parenthesesCounter -=1
                    if parenthesesCounter == 0:
                        lineRemovingOn = False
                    # Put the correct End Sites
                    if(parenthesesCounter==2 and "OFFSET" in line.split(" ") and (jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex)):
                        # manual correction to add the necessary 6 channels
                        if(jointCounter == rightHandIndex or jointCounter == leftHandIndex):
                            newHeader = newHeader[:-42]
                            newHeader += "\n"
                            newHeader += "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
                            newHeader += "\n"
                        newHeader += "End Site"
                        newHeader += "\n"
                        newHeader += "{"
                        newHeader += "\n"
                        newEndSite = f"OFFSET " + line.split(" ")[1] + " " + line.split(" ")[2] + " " + line.split(" ")[3]
                        newHeader += newEndSite
                        newHeader += "\n"
                        newHeader += "}"
                        newHeader += "\n"
                if not lineRemovingOn: 
                    newHeader += line
                    newHeader += "\n"
            # remove the last \n from the new header
            newHeader = newHeader[:-1]
            # reformat the data (remove the unused numbers)
            data = data.tolist()
            for index in range(0, len(data)):
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersEneko, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")
               
    # # main agent
    # for filename in sorted(os.listdir(nameBvhMainAgent)):
    #     bvhFilename = os.path.join(nameBvhMainAgent, filename)
    #     writeBvhFilename = os.path.join(writeDirPathMainAgent, filename)
    #     lineRemovingOn = False
    #     # checking if it is a file
    #     print(bvhFilename)
    #     if os.path.isfile(bvhFilename):
    #         header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
    #         newHeader = ""
    #         # reformat the header (remove the unwanted bones)
    #         jointCounter = 0
    #         parenthesesCounter = 0
    #         for line in header.split("\n"):
    #             if "JOINT" in line.split(" "):
    #                 jointCounter += 1
    #             if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
    #                 if "JOINT" in line.split(" "):
    #                     parenthesesCounter +=1
    #                 lineRemovingOn = True
    #             if lineRemovingOn:
    #                 if "{" in line.split(" "):
    #                     parenthesesCounter +=1
    #                 if "}" in line.split(" "):
    #                     parenthesesCounter -=1
    #                 if parenthesesCounter == 0:
    #                     lineRemovingOn = False
    #             if not lineRemovingOn: 
    #                 newHeader += line
    #                 newHeader += "\n"
    #         # remove the last \n from the new header
    #         newHeader = newHeader[:-1]
    #         # reformat the data (remove the unused numbers)
    #         data = data.tolist()
    #         for index in range(0, len(data)):
    #             data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

    #         with open(writeBvhFilename, "w") as f:
    #             f.write(newHeader)
    #             for row in data:
    #                 f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
    #                 f.write("\n")


    # validation split processing old filename
    nameBvhInterlocutor = "../../enekoDataset/genea2023_val/"
    writeDirPathInterlocutor = "../../enekoDatasetNoHands/genea2023_val/"

    # nameBvhMainAgent = "../../silenceDataset3sec/genea2023_val/genea2023_dataset/val/main-agent/bvh/"
    # writeDirPathMainAgent = "../../silenceDataset3secNoHands/genea2023_val/genea2023_dataset/val/main-agent/bvh/"

    if not os.path.exists(writeDirPathInterlocutor): os.makedirs(writeDirPathInterlocutor)
    # if not os.path.exists(writeDirPathMainAgent): os.makedirs(writeDirPathMainAgent)

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
                data[index] = np.delete(data[index], faceAndHandsIndexesNumbersEneko, axis=0).copy()

            with open(writeBvhFilename, "w") as f:
                f.write(newHeader)
                for row in data:
                    f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                    f.write("\n")
            
    # # main agent
    # for filename in sorted(os.listdir(nameBvhMainAgent)):
    #     bvhFilename = os.path.join(nameBvhMainAgent, filename)
    #     writeBvhFilename = os.path.join(writeDirPathMainAgent, filename)
    #     lineRemovingOn = False
    #     # checking if it is a file
    #     print(bvhFilename)
    #     if os.path.isfile(bvhFilename):
    #         header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
    #         newHeader = ""
    #         # reformat the header (remove the unwanted bones)
    #         jointCounter = 0
    #         parenthesesCounter = 0
    #         for line in header.split("\n"):
    #             if "JOINT" in line.split(" "):
    #                 jointCounter += 1
    #             if jointCounter == faceIndex or jointCounter == rightHandIndex or jointCounter == leftHandIndex:
    #                 if "JOINT" in line.split(" "):
    #                     parenthesesCounter +=1
    #                 lineRemovingOn = True
    #             if lineRemovingOn:
    #                 if "{" in line.split(" "):
    #                     parenthesesCounter +=1
    #                 if "}" in line.split(" "):
    #                     parenthesesCounter -=1
    #                 if parenthesesCounter == 0:
    #                     lineRemovingOn = False
    #             if not lineRemovingOn: 
    #                 newHeader += line
    #                 newHeader += "\n"
    #         # remove the last \n from the new header
    #         newHeader = newHeader[:-1]
    #         # reformat the data (remove the unused numbers)
    #         data = data.tolist()
    #         for index in range(0, len(data)):
    #             data[index] = np.delete(data[index], faceAndHandsIndexesNumbersGenea, axis=0).copy()

    #         with open(writeBvhFilename, "w") as f:
    #             f.write(newHeader)
    #             for row in data:
    #                 f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
    #                 f.write("\n")