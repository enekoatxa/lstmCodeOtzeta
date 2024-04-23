import torchvision
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader

class sequencesDataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, datasetName = "silenceDataset3sec", partition = "Validation", isTiny = False, specificSize = -1, trim = False, sequenceSize = -1, verbose = False, specificTrim = -1, outSequenceSize=1):
        """
        :param datasetName: name of the dataset that we want to load.
        :param partition: name of the partition that we want to load.
        :param is_tiny: for testing purposes
        """

        # galdera: superri dei hau zertarako da?
        super(sequencesDataset, self).__init__(root + "/" + datasetName)
        
        # load 
        x, y, ids = bvhLoader.loadSequenceDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, trim=trim, sequenceSize=sequenceSize, verbose=verbose, specificTrim=specificTrim, outSequenceSize=outSequenceSize)
        
        # isTiny
        if isTiny:
            x = x[:10]
            y = y[:10]
            ids = ids[:10]

        self.sequences = x
        self.results = y
        self.ids = ids
        self.length = len(x)

        with open(1, "w", closefd=False) as f:
            print("dataloader length:" + str(self.length), file=f, flush=True)

    def __getitem__(self, index):
        """
        :param index: index of the person
        :return: tuple(list of all sequences in a bvh file, list of all results for all sequences, id of the person) 
        """
        sequences = self.sequences[index]
        results = self.results[index]
        ids = self.ids[index]

        return sequences, results, ids

    def __len__(self):
        return self.length