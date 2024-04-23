# multivariate output stacked lstm example
import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
from lstmDataset import lstmDataset
# import plotly.express as px
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# choose a number of time steps (sequence size)
n_steps = 20
n_steps_out = 10
def train():
    # load first datamodule
    datamodulePos = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, partition="Train", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, onlyPositions=True, outSequenceSize=n_steps_out)
    n_features = len(datamodulePos.sequences[0][0]) # datamodulePos.sequences[0][0] is a vector, of dimension 249
    # define model for positions
    model = Sequential()
    model.add(LSTM(1, activation='relu', input_shape=(n_steps, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(1, activation='relu', return_sequences=True))
    model.add((Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # fit model
    historyPos = model.fit(datamodulePos, epochs=1, verbose=1)

    model.save("models/positionsLSTM.keras")
    del datamodulePos

    # load second datamodule
    datamoduleRot = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, partition="Train", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, onlyRotations=True, outSequenceSize=n_steps_out)
    # define model for rotations
    model = Sequential()
    model.add(LSTM(1, activation='relu', input_shape=(n_steps, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(1, activation='relu', return_sequences=True))
    model.add((Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    historyRot = model.fit(datamoduleRot, epochs=1, verbose=1)

    model.save("models/rotationsLSTM.keras")
    del datamoduleRot
    # plot history
    # plt.style.use('science')
    plt.plot(historyPos.history['loss'])
    plt.plot(historyRot.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['pos', 'rot'], loc='upper left')
    plt.show()

def test():
    # demonstrate prediction
    modelPos = load_model("models/positionsLSTM.keras")
    modelRot = load_model("models/rotationsLSTM.keras")
    x_input_pos, x_input_rot, y_pos, y_rot, ids = bvhLoader.loadSequenceDatasetSeparateRotationsAndPositions(datasetName="silenceDataset3sec", partition="Train", verbose=True, specificSize=10, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out)
    #########################
    # predict the positions #
    #########################
    x_input_pos = x_input_pos[0]
    x_input_pos = np.array(x_input_pos)
    x_input_pos = x_input_pos.reshape((1, n_steps, 249))
    newXpos = modelPos.predict(x_input_pos, verbose=0)
    finalOutputPos = []
    finalOutputPos = np.append(x_input_pos[0], newXpos, axis=0)
    # prediction loop
    steps = 100
    for step in range(0, steps):
        x_input_pos = x_input_pos.reshape(n_steps, 249)
        x_input_pos = np.append(x_input_pos, newXpos, axis=0)
        x_input_pos = np.delete(x_input_pos, 0, axis=0)
        x_input_pos = x_input_pos.reshape((1, n_steps, 249))
        newXpos = modelPos.predict(x_input_pos, verbose=0)
        finalOutputPos = np.append(finalOutputPos, newXpos, axis=0)

    #########################
    # predict the rotations #
    #########################
    x_input_rot = x_input_rot[0]
    x_input_rot = np.array(x_input_rot)
    x_input_rot = x_input_rot.reshape((1, n_steps, 249))
    newXrot = modelRot.predict(x_input_rot, verbose=0)
    finalOutputRot = []
    finalOutputRot = np.append(x_input_rot[0], newXrot, axis=0)
    # prediction loop
    steps = 100
    for step in range(0, steps):
        x_input_rot = x_input_rot.reshape(n_steps, 249)
        x_input_rot = np.append(x_input_rot, newXrot, axis=0)
        x_input_rot = np.delete(x_input_rot, 0, axis=0)
        x_input_rot = x_input_rot.reshape((1, n_steps, 249))
        newXrot = modelRot.predict(x_input_rot, verbose=0)
        finalOutputRot = np.append(finalOutputRot, newXrot, axis=0)

    with open("testBvh.bvh", "w") as f:
        for linePos, lineRot in zip(finalOutputPos, finalOutputRot):
            for index in range(0, len(linePos)-2, 3):
                f.write(str(linePos.tolist()[index]) + " " + str(linePos.tolist()[index+1]) + " " + str(linePos.tolist()[index+2]))
                f.write(" ")
                f.write(str(lineRot.tolist()[index]) + " " + str(lineRot.tolist()[index+1]) + " " + str(lineRot.tolist()[index+2]))
                f.write(" ")
            f.write("\n")
            # f.write(str(linePos.tolist()[index]) + " " + str(linePos.tolist()[index+1]) + " " + str(linePos.tolist()[index+2]))
            # f.write(str(lineRot.tolist()[index]) + " " + str(lineRot.tolist()[index+1]) + " " + str(lineRot.tolist()[index+2]))        
            f.close


def testMultiple():
    # demonstrate prediction
    modelPos = load_model("models/positionsLSTM.keras")
    modelRot = load_model("models/rotationsLSTM.keras")
    x_input_pos, x_input_rot, y_pos, y_rot, ids = bvhLoader.loadSequenceDatasetSeparateRotationsAndPositions(datasetName="silenceDataset3sec", 
        partition="Train", verbose=True, specificSize=10, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out)
    #########################
    # predict the positions #
    #########################
    x_input_pos = x_input_pos[0]
    x_input_pos = np.array(x_input_pos)
    x_input_pos = x_input_pos.reshape((1, n_steps, 249))
    newXpos = modelPos.predict(x_input_pos, verbose=0)
    finalOutputPos = []
    # prediction loop
    newXpos = modelPos.predict(x_input_pos, verbose=0)
    x_input_pos = x_input_pos.reshape((n_steps, 249))
    newXpos = newXpos.reshape((n_steps_out, 249))
    finalOutputPos = np.append(x_input_pos, newXpos, axis=0)

    #########################
    # predict the rotations #
    #########################
    x_input_rot = x_input_rot[0]
    x_input_rot = np.array(x_input_rot)
    x_input_rot = x_input_rot.reshape((1, n_steps, 249))
    newXrot = modelRot.predict(x_input_rot, verbose=0)
    finalOutputRot = []
    # prediction loop
    newXrot = modelRot.predict(x_input_rot, verbose=0)
    x_input_rot = x_input_rot.reshape((n_steps, 249))
    newXrot = newXrot.reshape((n_steps_out, 249))
    finalOutputRot = np.append(x_input_rot, newXrot, axis=0)

    with open("testBvh.bvh", "w") as f:
        for linePos, lineRot in zip(finalOutputPos, finalOutputRot):
            for index in range(0, len(linePos)-2, 3):
                f.write(str(linePos.tolist()[index]) + " " + str(linePos.tolist()[index+1]) + " " + str(linePos.tolist()[index+2]))
                f.write(" ")
                f.write(str(lineRot.tolist()[index]) + " " + str(lineRot.tolist()[index+1]) + " " + str(lineRot.tolist()[index+2]))
                f.write(" ")
            f.write("\n")
            # f.write(str(linePos.tolist()[index]) + " " + str(linePos.tolist()[index+1]) + " " + str(linePos.tolist()[index+2]))
            # f.write(str(lineRot.tolist()[index]) + " " + str(lineRot.tolist()[index+1]) + " " + str(lineRot.tolist()[index+2]))        
            f.close

if __name__ == "__main__":
    train()
    testMultiple()
