from vaeDataset import vaeDataset
from vaeDataloader import vaeDataloader
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
    args = argsClass(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, numWorkers = 6, modelName="gpt", datasetName = "silenceDataset3sec")
    datamodule = vaeDataloader(args)
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)
    print(len(batch))
    print(len(batch[0]))

if __name__ == "__main__":
    main()