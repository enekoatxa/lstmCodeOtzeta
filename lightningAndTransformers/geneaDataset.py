import torchvision
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader

class geneaDataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, datasetName = "silenceDataset3sec", partition = "Validation", isTiny = False, specificSize = -1, verbose = False, trim = False, specificTrim = -1):
        """
        :param datasetName: name of the dataset that we want to load.
        :param partition: name of the partition that we want to load.
        :param isTiny: for testing purposes
        """

        # galdera: superri dei hau zertarako da?
        super(geneaDataset, self).__init__(root + "/" + datasetName)
        
        # load 
        x, y, ids = bvhLoader.loadDatasetAndCreateResults(datasetName, partition = partition, specificSize = specificSize, verbose = verbose, trim = trim, specificTrim = specificTrim)
        
        if isTiny:
            x = x[:10]
            y = y[:10]
            ids = ids[:10]

        self.frames = x
        self.results = y
        self.ids = ids
        self.length = len(x)
        
        with open(1, "w", closefd=False) as f:
            print("dataloader length:" + str(self.length), file=f, flush=True)

    def __getitem__(self, index):
        """
        :param index: index of the person
        :return: tuple(list of all frames in a bvh file, list of all results for all frames, ids of the person) 
        """
        frames = self.frames[index]
        results = self.results[index]
        # TODO: add the person ids
        ids = self.ids[index]

        return frames, results, ids

    def __len__(self):
        return self.length