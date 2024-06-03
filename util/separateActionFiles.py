
import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import numpy as np
import csv
# train split processing old filename
actionsFolder = "../../enekoDatasetNoHandsCen2/actions/"
outputActionsFolder = "../../enekoDatasetNoHandsCen2/actionsSeparated/"
annotationsCsvPath = "../../enekoDatasetNoHandsCen2/actions_banaketa.csv"
if not os.path.exists(outputActionsFolder): os.makedirs(outputActionsFolder)

# read the csv
annotationsDict = dict()
with open(annotationsCsvPath, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        if(row[0]!="Actions"):
            annotationsDict[row[0]] = row[1:]


for filename in sorted(os.listdir(actionsFolder)):
    bvhFilename = os.path.join(actionsFolder, filename)
    print(bvhFilename)
    # checking if it is a file
    personIndex = (os.path.basename(bvhFilename).split("_")[0])
    if os.path.isfile(bvhFilename):
        header, data = bvhLoader.loadBvhToList(bvhFilename, returnHeader=True, returnData=True, returnCounter=False, reorderVectors=False)
        header = header.replace("\t", "")

        # start separating actions
        lastCutPointFrame = 0
        for cutPointIndex in range(0, len(annotationsDict[personIndex])):
            cutPointFrame = annotationsDict[personIndex][cutPointIndex]
            writeBvhFilename = os.path.join(outputActionsFolder, personIndex + "_" + str(int(cutPointIndex)) + "_" + annotationsDict[personIndex + " Actions"][cutPointIndex] + ".bvh")
            if(annotationsDict[personIndex + " Actions"][cutPointIndex]!="OKERRA"):
                with open(writeBvhFilename, "w") as f:
                    f.write(header)
                    for row in data[int(lastCutPointFrame):int(cutPointFrame)]:
                        f.write(str(row.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                        f.write("\n")
                    lastCutPointFrame = cutPointFrame