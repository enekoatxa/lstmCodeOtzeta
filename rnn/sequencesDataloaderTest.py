from sequencesDataloader import sequencesDataloader
import numpy as np
#TODO: PROBATU EA DATALOADERRA ONDO DABILEN, ETA ZERBAIT KARGATZEN DUEN
class argsClass():
    def __init__(self, root, isTiny, batchSize, numWorkers, modelName, datasetName, sequenceSize=1, trim=False, specificTrim=-1, specificSize=-1, verbose = False):
        self.root = root
        self.isTiny = isTiny
        self.batchSize = batchSize
        self.numWorkers = numWorkers
        self.modelName  = modelName
        self.datasetName = datasetName
        self.sequenceSize = sequenceSize
        self.trim = trim
        self.specificTrim = specificTrim
        self.specificSize = specificSize
        self.verbose = verbose

def main():
    args = argsClass(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, numWorkers = 6, modelName="roberta", datasetName = "silenceDataset3sec", sequenceSize= 10, trim=False, specificSize=200)
    datamodule = sequencesDataloader(args)
    batch = next(iter(datamodule.train_dataloader()))
    print(np.shape(batch[0])) #batch[0]k sekuentziak dauzka, batch[1]ek emaitzak, batch[2]k id ak
    batch = next(iter(datamodule.train_dataloader()))
    print(batch)

if __name__ == "__main__":
    main()