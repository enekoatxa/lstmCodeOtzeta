import numpy as np
from torch.utils import data
from geneaDataset import geneaDataset
import pytorch_lightning as pl

class geneaDataloader(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.root = args.root
        self.isTiny = args.isTiny
        self.batchSize = args.batchSize
        self.numWorkers = args.numWorkers
        self.modelname = args.modelname
        self.datasetName = args.datasetName
        self.verbose = args.verbose
        self.trim = args.trim
        self.specificTrim = args.specificTrim
        self.specificSize = args.specificSize

    def train_dataloader(self):
        # the dataloader should load depending on specifications datasetName and modelname
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = geneaDataset(root = self.root, datasetName=self.datasetName, partition = "Train", isTiny=self.isTiny, verbose=self.verbose, trim = self.trim, specificSize=self.specificSize, specificTrim=self.specificTrim)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)
    
    def val_dataloader(self):
        if self.datasetName == "silenceDataset3sec" or self.datasetName == "silenceDataset2sec" or self.datasetName == "silenceDataset1sec":
            dataset = geneaDataset(root = self.root, datasetName=self.datasetName, partition = "Validation", isTiny=self.isTiny)

        params = {
            'batch_size': self.batchSize,
            'shuffle': True,
            'num_workers': self.numWorkers
        }
        return data.DataLoader(dataset, **params)

    def test_dataloader(self):
        print("Test partition has not been implemented. aborting dataset loader...")
        return