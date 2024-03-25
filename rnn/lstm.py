# multivariate output stacked lstm example
import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam, SGD
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
from lstmDataset import lstmDataset
# import plotly.express as px
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from pickle import dump, load
from keras import backend as K
from keras import ops

# choose a number of time steps (sequence size)
n_steps = 500 # number of frames of the input sequence
n_steps_out = 1 # number of frames of the output sequence (now I always use 1)
n_features = 0 # number of features of each input vector (assigned later in code)
''' Don't use
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(self.model.optimizer.learning_rate)
        testDifferences("models/callbackModel.keras", 0, self.model)

def my_loss_fn(y_true, y_pred):
    squared_difference = ops.square(y_true - y_pred)
    return ops.mean(squared_difference, axis=-1)  # Note the `axis=-1`

def gaussian_nll(ytrue, ypreds):
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)
'''
def train(useDifferences, jump):
    ###################################################
    ### ORIGINAL TRAINING METHOD: MODELLING VECTORS ###
    ###################################################
    if not useDifferences:
        # scaler = bvhLoader.createAndFitStandardScaler(datasetName = "silenceDataset3sec", removeHandsAndFace=True)
        # dump(scaler, open("scaler.pkl", "wb"))
        scaler = load(open("scaler.pkl", "rb"))
    else:
        ##################################################
        ### NEW TRAINING METHOD: MODELLING DIFFERENCES ###
        ##################################################
        #scaler = bvhLoader.createAndFitStandardScalerForDifferences(datasetName = "silenceDataset3sec", removeHandsAndFace=True)
        #dump(scaler, open("scalerDifferences.pkl", "wb"))
        scaler = load(open("scalerDifferences" + str(jump) + ".pkl", "rb"))
        # scalerRot = load(open("scalerDifferences" + str(jump) + "onlyRotations.pkl", "rb"))

    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Train", datasetName = "dataset", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = useDifferences, jump = jump)
    datamoduleVal = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Validation", datasetName = "dataset", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = useDifferences, jump = jump)
    
    global n_features
    n_features = len(datamodule.sequences[0][0]) # datamodule.sequences[0][0] is a vector, of dimension n_features
    ####################
    # DEFINE THE MODEL #
    ####################
    
    learning_rate = 0.0001
    # I still haven't tested this model
    '''model = Sequential()
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(rate = 0.2)) # hurrengo lerroan input_shape=(n_steps, n_features) da berez
    model.add(LSTM(1000, activation = 'tanh', input_shape=(n_steps, 500), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    # opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    '''
    # Model that doesn't use the time distributed layer (generates 1 output vector). Test it with the test() function
    model = Sequential()
    model.add(LSTM(1000, activation = 'tanh', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    # opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    history = model.fit(datamodule, validation_data=datamoduleVal, epochs=500, verbose=1)#, callbacks = [CustomCallback()])
    
    model.save("models/differencesDataset" + str(jump) + ".keras")

    # save the training variables in a plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("resultImages/trainingLoss.png")
    plt.close()
    
    # apply early stopping to the network training
    # early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 8, verbose = 1)

######################################################################
### use this method to generate the animations ONE FRAME AT A TIME ###
######################################################################
        
def testDifferences(modelName, jump, preparedModel):
    # demonstrate prediction
    if(modelName!="models/callbackModel.keras"):
        model = load_model(modelName) # "models/singleLSTMNoLimbsOneFrame.keras"
    else:
        model = preparedModel
    scaler = load(open("scalerDifferences" + str(jump) + ".pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, sequenceSize = n_steps, specificSize=1, trim=False, removeHandsAndFace=False, scaler=scaler, loadDifferences = True, jump=jump)
    finalOutput = []
    x_input = x_input[0]
    x_input = scaler.inverse_transform(x_input)
    lastPosition = firstPersonPositions[0]
    # prepare the input
    finalOutput.append((x_input[0] + lastPosition))
    lastPosition = x_input[0] + lastPosition
    for vector in range(1, len(x_input)):
        finalOutput.append((x_input[vector] + lastPosition))
        lastPosition = (finalOutput[-1]) # new last position is the last added vector
    x_input = x_input.reshape((1, n_steps, n_features))
    newX = model.predict(x_input, verbose=0)
    newX = scaler.inverse_transform(newX)
    lastPosition = newX + lastPosition
    finalOutput = np.append(finalOutput, lastPosition, axis=0)
    # prediction loop
    steps = 500
    for step in range(0, steps-1):
        print(step)
        x_input = x_input.reshape(n_steps, n_features)
        x_input = np.append(x_input, newX, axis=0)
        x_input = np.delete(x_input, 0, axis=0)
        x_input = x_input.reshape((1, n_steps, n_features))
        newX = model.predict(x_input, verbose=0)
        newX = scaler.inverse_transform(newX)
        lastPosition = newX + lastPosition
        finalOutput = np.append(finalOutput, lastPosition.copy(), axis=0)

    # finalOutput = scaler.inverse_transform(finalOutput)

    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close

##########################################################################
### use this method to generate the animations BY GENERATING SEQUENCES ### not used currently
##########################################################################
        
def testDifferencesMultiple(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/multipleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scalerDifferences.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, specificSize=200, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, removeHandsAndFace=True, scaler = scaler, loadDifferences = True)
    #######################
    # predict the changes #
    #######################
    x_input = x_input[0]
    lastPosition = firstPersonPositions[n_steps]
    x_input = np.array(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))
    finalOutput = []
    # prediction loop
    newX = model.predict(x_input, verbose=0)
    x_input = x_input.reshape((n_steps, n_features))
    newX = newX.reshape((n_steps_out, n_features))
    # we have the changes in new_X, now add the initial vector, and the last vector each step
    for vec in newX:
        vec = lastPosition + vec
        lastPosition = vec

    finalOutput = np.append(x_input, newX, axis=0)
    finalOutput = scaler.inverse_transform(finalOutput)
    print(finalOutput.shape)
    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close
'''
#############
# DON'T USE #
#############
def test(modelName, preparedModel = None):
    # demonstrate prediction
    if(modelName!="models/callbackModel.keras"):
        model = load_model(modelName) # "models/singleLSTMNoLimbsOneFrame.keras"
    else:
        model = preparedModel
    scaler = load(open("scaler.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="dataset", partition="Validation", verbose=True, specificSize=3, trim=True, sequenceSize=n_steps, removeHandsAndFace=True, scaler=scaler)
    x_input = x_input[0]
    x_input = x_input.reshape((1, n_steps, n_features))
    newX = model.predict(x_input, verbose=0)
    finalOutput = []
    finalOutput = np.append(x_input[0], newX, axis=0)
    # prediction loop
    steps = 50
    for step in range(0, steps-1):
        x_input = x_input.reshape(n_steps, n_features)
        x_input = np.append(x_input, newX, axis=0)
        x_input = np.delete(x_input, 0, axis=0)
        x_input = x_input.reshape((1, n_steps, n_features))
        newX = model.predict(x_input, verbose=0)
        finalOutput = np.append(finalOutput, newX, axis=0)
    finalOutput = scaler.inverse_transform(finalOutput)
    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close

#############
# DON'T USE #
#############
def testMultiple(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/multipleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scaler.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, specificSize=200, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, removeHandsAndFace=True, scaler = scaler, loadDifferences = False)
    #########################
    # predict the positions #
    #########################
    x_input = x_input[0]
    x_input = np.array(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))
    finalOutput = []
    # prediction loop
    newX = model.predict(x_input, verbose=0)
    x_input = x_input.reshape((n_steps, n_features))
    newX = newX.reshape((n_steps_out, n_features))
    finalOutput = np.append(x_input, newX, axis=0)
    finalOutput = scaler.inverse_transform(finalOutput)
    print(finalOutput.shape)
    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close

#############
# DON'T USE #
#############
def gluePositionAndRotation(position, rotation):
    ret = []
    for posIndex in range(0, len(position), 3):
        ret.append(position[posIndex])
        ret.append(position[posIndex+1])
        ret.append(position[posIndex+2])
        ret.append(rotation[posIndex])
        ret.append(rotation[posIndex+1])
        ret.append(rotation[posIndex+2])
    return ret

#############
# DON'T USE #
#############
def testDifferencesAnglesAndPositions(modelNamePos, modelNameRot, jump):
    # demonstrate prediction
    modelPos = load_model(modelNamePos) # "models/singleLSTMNoLimbsOneFrame.keras"
    modelRot = load_model(modelNameRot) # "models/singleLSTMNoLimbsOneFrame.keras"
    scalerPos = load(open("scalerDifferences" + str(jump) + "onlyPositions.pkl", "rb"))
    scalerRot = load(open("scalerDifferences" + str(jump) + "onlyRotations.pkl", "rb"))
    x_input_pos, y, firstPersonPositions_pos, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Train", verbose=True, sequenceSize = n_steps, specificSize=1000, trim=False, removeHandsAndFace=True, scaler=scalerPos, onlyPositions=True, loadDifferences = True, jump=jump)
    x_input_rot, y, firstPersonPositions_rot, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Train", verbose=True, sequenceSize = n_steps, specificSize=1000, trim=False, removeHandsAndFace=True, scaler=scalerRot, onlyRotations=True, loadDifferences = True, jump=jump)
    x_input_pos = x_input_pos[100]
    x_input_rot = x_input_rot[100]
    x_input_pos = scalerPos.inverse_transform(x_input_pos)
    x_input_rot = scalerRot.inverse_transform(x_input_rot)
    lastPosition = gluePositionAndRotation(firstPersonPositions_pos[100], firstPersonPositions_rot[100])
    
    # prediction loop variables
    steps = 100
    finalOutputPos = []
    finalOutputRot = []
    finalOutputDiff = []
    finalOutput = []

    for pos, rot in zip(x_input_pos, x_input_rot):
        finalOutputDiff.append(gluePositionAndRotation(pos, rot))

    # first prediction out of the for loop
    x_input_pos = x_input_pos.reshape((1, n_steps, n_features))
    x_input_rot = x_input_rot.reshape((1, n_steps, n_features))
    newX_pos = modelPos.predict(x_input_pos, verbose=0)
    newX_rot = modelRot.predict(x_input_rot, verbose=0)
    newX_pos = scalerPos.inverse_transform(newX_pos)
    newX_rot = scalerRot.inverse_transform(newX_rot)

    for step in range(0, steps-1):
        newX_pos = newX_pos.reshape(1, n_features)
        newX_rot = newX_rot.reshape(1, n_features)
        x_input_pos = x_input_pos.reshape(n_steps, n_features)
        x_input_rot = x_input_rot.reshape(n_steps, n_features)
        x_input_pos = np.append(x_input_pos, newX_pos, axis=0)
        x_input_rot = np.append(x_input_rot, newX_rot, axis=0)
        x_input_pos = np.delete(x_input_pos, 0, axis=0)
        x_input_rot = np.delete(x_input_rot, 0, axis=0)
        x_input_pos = x_input_pos.reshape((1, n_steps, n_features))
        x_input_rot = x_input_rot.reshape((1, n_steps, n_features))
        newX_pos = modelPos.predict(x_input_pos, verbose=0)
        newX_rot = modelRot.predict(x_input_rot, verbose=0)
        newX_pos = scalerPos.inverse_transform(newX_pos)
        newX_rot = scalerRot.inverse_transform(newX_rot)
        newX_pos = newX_pos.reshape(n_features)
        newX_rot = newX_rot.reshape(n_features)
        finalOutputPos.append(newX_pos.copy())
        finalOutputRot.append(newX_rot.copy())

    for pos, rot in zip(finalOutputPos, finalOutputRot):
        finalOutputDiff.append(gluePositionAndRotation(pos, rot))

    for diff in finalOutputDiff:
        lastPosition = np.add(np.asarray(lastPosition), np.asarray(diff))
        finalOutput.append(lastPosition.tolist())

    with open("resultBvhs/" + modelNamePos.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close
'''
# method to check if the data is loading correctly
def checkDatamodule():
    # convert into input/output
    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, partition="Train", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, specificSize=10, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, loadDifferences=True)
    
    print(len(datamodule[0][0][0]))
    print(datamodule[0][0][0])
    print(datamodule[0][0][1])
    print(datamodule[0][0][2])
    print(datamodule[0][0][3])
    print(datamodule[0][0][4])
    print(len(datamodule[0][1][0]))

# method to check if a specific model is created correctly (prints out the summary of a model I'm testing)
def checkNetwork():
    model = Sequential()
    model.add(LSTM(200, activation = 'relu', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    # model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation = 'relu', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add((Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

def visualizeWeights():
    model = load_model("models/multipleLSTMNoLimbsOneFrame.keras")
    layer = model.layers[0]
    print(layer)
    weights = layer.get_weights()
    print(weights)

def prepareScalerForJump(jump, extraInfo, datasetName = "silenceDataset3sec"):
    onlyPos = extraInfo=="onlyPositions"
    onlyRot = extraInfo=="onlyRotations"
    scaler = bvhLoader.createAndFitStandardScalerForDifferences(datasetName = datasetName, specificSize = 100, removeHandsAndFace=True, jump=jump, onlyPositions=onlyPos, onlyRotations=onlyRot)
    dump(scaler, open("scalerDifferences" + str(jump) + str(extraInfo) + ".pkl", "wb"))

if __name__ == "__main__":
    # create the standard scaler specifically for the data that we are going to use (just execute once)     
    prepareScalerForJump(0, "", "dataset")
    # train the model
    train(useDifferences = True, jump=0)
    # test the model
    testDifferences("models/differencesDatasetEneko0.keras", jump=0, preparedModel=None)
    # ALL THESE METHODS ARE COMMENTED (I've used them during development to test many things and to make the inference in many ways)
    # testDifferencesAnglesAndPositions("models/differencesNoLimbsOneFramePosJump0.keras","models/differencesNoLimbsOneFrameRotJump0.keras", jump=0)
    # testDifferences("models/differencesDatasetEneko0.keras", jump=0, preparedModel=None)
    # test("models/differencesDataset0.keras")
    # testDifferencesMultiple("models/differencesNoLimbsSomeFrames.keras")
    # test("models/originalModel.keras")
    # test("models/multipleLSTMNoLimbsKldOneFrame.keras")
    # test("models/multipleLSTMNoLimbsMapeOneFrame.keras")
    # test("models/multipleLSTMNoLimbsMsleOneFrame.keras")
    # test("models/multipleLSTMNoLimbsMseOneFrame.keras")
    # checkNetwork()
    # visualizeWeights()
    # checkDatamodule()