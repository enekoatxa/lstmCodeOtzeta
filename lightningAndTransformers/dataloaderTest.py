from geneaDataloader import geneaDataloader
from geneaDataset import geneaDataset
#TODO: PROBATU EA DATALOADERRA ONDO DABILEN, ETA ZERBAIT KARGATZEN DUEN
class argsClass():
    def __init__(self, root, isTiny, batchSize, numWorkers, modelName, datasetName):
        self.root = root
        self.isTiny = isTiny
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.modelName  = modelName
        self.datasetName = datasetName

def main():
    # dataset =  geneaDataset(root="/home/bee/Desktop/idle animation generator", datasetName="silenceDataset3sec", partition = "Train", isTiny=False)
    # dataset2 = geneaDataset(root="/home/bee/Desktop/idle animation generator", datasetName="silenceDataset3sec", partition = "Validation", isTiny=False)

    # print("Number of rows:" + str(len(dataset)))
    # print("Number of rows:" + str(len(dataset2)))
    # print("frames of person 0 :" + str((dataset[0][0])))
    # print("frames of person 0 :" + str((dataset2[0][0])))
    # print("results of person 0 :" + str((dataset[0][1])))
    # print("results of person 0 :" + str((dataset2[0][1])))
    # print("ids of person 0 :" + str((dataset[0][2])))
    # print("ids of person 0 :" + str((dataset2[0][2])))
    
    args = argsClass(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, numWorkers = 6, modelName="roberta", datasetName = "silenceDataset3sec")
    datamodule = geneaDataloader(args)
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)

if __name__ == "__main__":
    main()