import torchvision
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
from sklearn.decomposition import PCA
import numpy as np

class vaeDataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, datasetName = "silenceDataset3sec", partition = "Validation", isTiny = False, verbose = False, specificSize = -1, pcaComp=-1):
        """
        :param datasetName: name of the dataset that we want to load.
        :param partition: name of the partition that we want to load.
        :param isTiny: for testing purposes
        """

        super(vaeDataset, self).__init__(root + "/" + datasetName)
        
        # load 
        x = bvhLoader.loadDatasetForVae(datasetName, partition=partition, verbose=verbose, specificSize=specificSize)

        # isTiny
        if isTiny:
            x = x[:10]

        # pca
        if pcaComp>-1:
            pca = PCA(n_components=pcaComp)
            x = pca.fit_transform(x)
            x = np.float32(x)

        self.frames = x
        self.length = len(x)

        with open(1, "w", closefd=False) as f:
            print("dataloader length:" + str(self.length), file=f, flush=True)

    def __getitem__(self, index):
        """
        :param index: index of the person
        :return: tuple(list of all frames in a bvh file, list of all results for all frames, id of the person) 
        """
        frames = self.frames[index]

        return frames

    def __len__(self):
        return self.length