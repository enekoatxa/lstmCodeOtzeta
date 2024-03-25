import numpy as np
from torch.utils import data
from sequencesDataset import sequencesDataset
import pytorch_lightning as pl

class sequencesDataloader(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.root = args.root
        self.isTiny =args.isTiny
        self.batchSize = args.batchSize
        self.numWorkers = args.numWorkers
        self.modelName = args.modelName
        self.datasetName = args.datasetName
        self.verbose = args.verbose
        # arguments to trim and change dataloader size
        self.trim = args.trim
        self.specificTrim = args.specificTrim
        self.specificSize = args.specificSize
        # sequences needed arguments
        self.sequenceSize = args.sequenceSize

    def train_dataloader(self):
        # the dataloader should load depending on specifications datasetName and modelName
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = sequencesDataset(root = self.root, datasetName=self.datasetName, partition="Train", isTiny=self.isTiny,
                                       specificSize=self.specificSize, trim=self.trim, sequenceSize=self.sequenceSize, verbose=self.verbose)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)
    
    def val_dataloader(self):
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = sequencesDataset(root = self.root, datasetName=self.datasetName, partition="Validation", isTiny=self.isTiny,
                                       specificSize=self.specificSize, trim=self.trim, sequenceSize=self.sequenceSize, verbose=self.verbose)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        print("Test partition has not been implemented. aborting dataset loader...")
        return