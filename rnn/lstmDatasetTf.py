import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
import numpy as np
import tensorflow as tf
class lstmDatasetTf():
    def __init__(self, root, datasetName = "silenceDataset3sec", partition = "Validation", isTiny = False, specificSize = -1, trim = False, sequenceSize = -1, verbose = False, specificTrim = -1, batchSize = 56, onlyPositions = False, onlyRotations = False, outSequenceSize=1, removeHandsAndFace = False, scaler = None, **kwargs):
        super().__init__(**kwargs)
        """
        :param datasetName: name of the dataset that we want to load.
        :param partition: name of the partition that we want to load.
        :param is_tiny: for testing purposes
        """

        # load everything
        x, y, ids = bvhLoader.loadSequenceDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, trim=trim,
                                                   sequenceSize=sequenceSize, verbose=verbose, specificTrim=specificTrim,
                                                   onlyPositions=onlyPositions, onlyRotations=onlyRotations, outSequenceSize=outSequenceSize, 
                                                   removeHandsAndFace = removeHandsAndFace, scaler = scaler)
        
        x = np.asarray(x)
        # x = np.expand_dims(x, 0)
        x = tf.convert_to_tensor(x)
        y = np.asarray(y)
        # y = np.expand_dims(y, 0)
        y = tf.convert_to_tensor(y)
        ids = np.asarray(ids)
        ids = tf.convert_to_tensor(ids)

        # isTiny
        if isTiny:
            x = x[:10]
            y = y[:10]
            ids = ids[:10]

        with open(1, "w", closefd=False) as f:
            print("dataloader length:" + str(len(x)), file=f, flush=True)
        self.batchSize = batchSize
        self.sequences = x
        self.results = y
        self.ids = ids
        self.scaler = scaler
        self.length = int(len(x)/self.batchSize)

    def getDataloader(self):
        return tf.data.Dataset.from_tensor_slices([self.sequences, self.results])