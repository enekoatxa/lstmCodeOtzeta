import numpy as np
from torch.utils import data
from vaeDataset import vaeDataset
import pytorch_lightning as pl

class vaeDataloader(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.root = args.root
        self.isTiny =args.isTiny
        self.batchSize = args.batchSize
        self.numWorkers = args.numWorkers
        self.modelName = args.modelName
        self.datasetName = args.datasetName
        self.specificSize = args.specificSize
        self.verbose = args.verbose

    def train_dataloader(self):
        # the dataloader should load depending on specifications datasetName and modelName
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = vaeDataset(root = self.root, datasetName=self.datasetName, partition = "Train", isTiny=self.isTiny, verbose = self.verbose, specificSize = self.specificSize)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)
    
    def val_dataloader(self):
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = vaeDataset(root = self.root, datasetName=self.datasetName, partition = "Validation", isTiny=self.isTiny, verbose = self.verbose, specificSize = self.specificSize)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        print("Test partition has not been implemented. aborting dataset loader...")
        return