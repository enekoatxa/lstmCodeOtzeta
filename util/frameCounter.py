import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader

baseFolderPath= "/home/bee/Desktop/idle animation generator/enekoDatasetNoHandsCen/"

allClasses = dict()
for dirpath, dirnames, filenames in sorted(os.walk(baseFolderPath)):
    for file in filenames:
        if(file.split(".")[-1]=='bvh'):
            if (file.split(".")[0].split("_")[-1] not in allClasses):
                allClasses[(file.split(".")[0].split("_")[-1])] = 0
            data =  bvhLoader.loadBvhToList(path = os.path.join(dirpath, file), returnCounter=False, returnData=True, returnHeader=False, returnMask=False)
            allClasses[(file.split(".")[0].split("_")[-1])] += len(data)

print(allClasses)