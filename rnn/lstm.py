# multivariate output stacked lstm example
import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam, SGD
import os
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
import bvhLoader
import quaternionsAndEulers
from lstmDataset import lstmDataset
# import plotly.express as px
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from pickle import dump, load
from keras import backend as K
from keras import ops
import math

# choose a number of time steps (sequence size)
n_steps = 80 # number of frames of the input sequence
n_steps_out = 1 # number of frames of the output sequence (now I always use 1)
n_features = 0 # number of features of each input vector (assigned later in code)
''' Don't use
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"{self.model.optimizer.learning_rate}")
        testDifferences("models/callbackModel.keras", 0, self.model)
'''
def my_loss_fn(y_true, y_pred):
    # MAE component of the loss function
    mae = np.abs(y_true - y_pred)
    # UNITARY QUATERNION component of the loss function
    y_true_rot, y_true_pos = quaternionsAndEulers.separateVectorsSimple(y_true, usingQuaternions=True)
    quat_loss = sum(((math.sqrt(y_true_rot[index] ** 2 + y_true_rot[index+1] ** 2 + y_true_rot[index+2] ** 2 + y_true_rot[index+3] ** 2) - 1) ** 2) for index in range(0, len(y_true_rot), 4))
    # quat_loss = 0
    # for index in range(0, len(y_true_rot)):
    #     if(index%4==0):
    #         norm = math.sqrt(y_true_rot[index] ** 2 + y_true_rot[index+1] ** 2 + y_true_rot[index+2] ** 2 + y_true_rot[index+3] ** 2)
    #         quat_loss = (norm - 1) ** 2 
    return ops.mean(mae, axis=-1) + quat_loss
'''
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
        # scaler = bvhLoader.createAndFitStandardScaler(datasetName = "silenceDataset3secNoHandsCen")
        # dump(scaler, open("scaler.pkl", "wb"))
        scaler = load(open("scalerQuaternionCen" + str(jump) + ".pkl", "rb"))
    else:
        ##################################################
        ### NEW TRAINING METHOD: MODELLING DIFFERENCES ###
        ##################################################
        #scaler = bvhLoader.createAndFitStandardScalerForDifferences(datasetName = "silenceDataset3secNoHandsCen")
        #dump(scaler, open("scalerDifferences.pkl", "wb"))
        scaler = load(open("scalerDifferencesQuaternion" + str(jump) + ".pkl", "rb"))
        # scalerRot = load(open("scalerDifferences" + str(jump) + "onlyRotations.pkl", "rb"))

    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Train", datasetName = "silenceDataset3secNoHandsCen", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, scaler = scaler, loadDifferences = useDifferences, jump = jump, useQuaternions = True)
    datamoduleVal = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Validation", datasetName = "silenceDataset3secNoHandsCen", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, scaler = scaler, loadDifferences = useDifferences, jump = jump, useQuaternions = True)
    
    global n_features
    n_features = len(datamodule.sequences[0][0]) # datamodule.sequences[0][0] is a vector, of dimension n_features
    ####################
    # DEFINE THE MODEL #
    ####################
    learning_rate = 0.0001
    # MODELO HONEK SKIP CONNECTIONA INPLEMENTATZEN DU
    input = keras.Input(shape=(80, n_features))
    inputSingle = keras.Input(shape=(1, n_features))
    lstm1 = LSTM(1000, activation = 'tanh', return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4)(input)
    lstm2 = LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4)(lstm1)
    dense = Dense(n_features)(lstm2)
    output = Add()([dense, inputSingle])
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    model = keras.Model(inputs=[input, inputSingle], outputs=output, name="skip_connection_model")
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    ''' MODELO HONEK BETI EMAITZA BERDINA EMATEN DU
    model = Sequential()
    model.add(Dense(100, input_shape=(n_steps, n_features), activation="relu"))
    model.add(Dropout(rate = 0.2))
    model.add(LSTM(1000, activation = 'tanh', input_shape=(n_steps, 100), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(rate = 0.2))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    # opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    '''
    '''
    # MODELO HONEK FUNTZIONATU DU
    # Model that doesn't use the time distributed layer (generates 1 output vector). Test it with the test() function
    model = Sequential()
    model.add(LSTM(1000, activation = 'tanh', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    # opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    '''

    history = model.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1)#, callbacks = [CustomCallback()])
    
    model.save("models/DatasetSkipConn" + str(jump) + ".keras")

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
    scalerEuler = load(open("scalerDifferencesEuler" + str(jump) + ".pkl", "rb"))
    scalerQuaternion = load(open("scalerDifferencesQuaternion" + str(jump) + ".pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Train", verbose=True, sequenceSize = n_steps, specificSize=1, trim=False, scaler=scalerEuler, loadDifferences = True, jump=jump, useQuaternions = False,  reorderVectors = True)
    finalOutput = []
    x_input = x_input[0]
    newx_input = []
    # convert the vectors to quaternions #
    # separate the vectors
    position = []
    rotation = []
    for index in range(0, len(x_input)):
        rotation, position = quaternionsAndEulers.separateVectorsSimple(x_input[index], usingQuaternions = False)
        rotation = quaternionsAndEulers.fromEulerToQuaternionVector(rotation)
        # after converting angles to quaternion, put the vectors back again, and process
        newx_input.append(quaternionsAndEulers.concatenateVectorsSimple(rotation, position, usingQuaternions = True))
    newx_input = scalerQuaternion.inverse_transform(newx_input)
    # create the first "latPosition" vector, so everything starts adding from this initial vector (have to convert it to quaternions)
    lastPosition = firstPersonPositions[0]
    lastPositionRot, lastPositionPos = quaternionsAndEulers.separateVectorsSimple(lastPosition, usingQuaternions = False)
    lastPositionRot = quaternionsAndEulers.fromEulerToQuaternionVector(lastPositionRot)
    lastPosition = quaternionsAndEulers.concatenateVectorsSimple(lastPositionRot, lastPositionPos, usingQuaternions = True)
    # if n_features is not set, set it using the quaternions + positions size
    global n_features
    if(n_features == 0):
        n_features = len(lastPosition)
    # prepare the input
    finalOutput.append((newx_input[0] + lastPosition))
    lastPosition = newx_input[0] + lastPosition
    for vector in range(1, len(newx_input)):
        finalOutput.append((newx_input[vector] + lastPosition))
        lastPosition = (finalOutput[-1]) # new last position is the last added vector
    newx_input = newx_input.reshape((1, n_steps, n_features))
    newX = model.predict(newx_input, verbose=0)
    newX = scalerQuaternion.inverse_transform(newX)
    lastPosition = newX + lastPosition
    finalOutput = np.append(finalOutput, lastPosition, axis=0)
    # prediction loop
    steps = 100
    for step in range(0, steps-1):
        print(f"inference step: {step}")
        newx_input = newx_input.reshape(n_steps, n_features)
        newx_input = np.append(newx_input, newX, axis=0)
        newx_input = np.delete(newx_input, 0, axis=0)
        newx_input = newx_input.reshape((1, n_steps, n_features))
        newX = model.predict(newx_input, verbose=0)
        newX = scalerQuaternion.inverse_transform(newX)
        lastPosition = newX + lastPosition
        finalOutput = np.append(finalOutput, lastPosition.copy(), axis=0)

    # finalOutput = scaler.inverse_transform(finalOutput)

    # pass every vector to euler angles
    intermediateRotation = []
    intermediatePosition = []
    finalEulerOutput = []
    for index in range(0, len(finalOutput)):
        intermediateRotation, intermediatePosition = quaternionsAndEulers.separateVectorsSimple(finalOutput[index], usingQuaternions = True)
        intermediateRotation = quaternionsAndEulers.fromQuaternionToEulerVector(intermediateRotation)
        finalEulerOutput.append(quaternionsAndEulers.glueVectors(intermediateRotation, intermediatePosition, vectorHeaderPath = "/home/bee/Desktop/idle animation generator/silenceDataset3secNoHandsCen/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/trn_2023_v0_000_interloctr_silence_0.bvh"))

    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalEulerOutput:
            f.write(str(line).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close

##########################################################################
### use this method to generate the animations BY GENERATING SEQUENCES ### not used currently
##########################################################################
        
def testDifferencesMultiple(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/multipleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scalerDifferencesEuler.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Validation", verbose=True, specificSize=200, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, scaler = scaler, loadDifferences = True, useQuaternions = True)
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
    print(f"{finalOutput.shape}")
    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close

def test(modelName, preparedModel = None):
    # demonstrate prediction
    if(modelName!="models/callbackModel.keras"):
        model = load_model(modelName)
    else:
        model = preparedModel
    scalerEuler = load(open("scalerEulerCen0.pkl", "rb"))
    scalerQuaternion = load(open("scalerQuaternionCen0.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Train", verbose=True, specificSize=40, trim=False, sequenceSize=n_steps, scaler=scalerQuaternion, useQuaternions = True, loadDifferences = False, reorderVectors = True)
    print(len(x_input))
    x_input = x_input[3000]

    # if n_features is not set, set it using the quaternions + positions size
    global n_features
    if(n_features == 0):
        n_features = len(x_input[0])
    x_input = x_input.reshape((1, n_steps, n_features))
    newX = model.predict(x_input, verbose=0)
    finalOutput = []
    finalOutput = np.append(x_input[0], newX, axis=0)
    
    # prediction loop
    steps = 1000
    for step in range(0, steps-1):
        x_input = x_input.reshape(n_steps, n_features)
        x_input = np.append(x_input, newX, axis=0)
        x_input = np.delete(x_input, 0, axis=0)
        x_input = x_input.reshape((1, n_steps, n_features))
        newX = model.predict(x_input, verbose=0)
        finalOutput = np.append(finalOutput, newX, axis=0)
    finalOutput = scalerQuaternion.inverse_transform(finalOutput)
    
    finalOutputOrdered = []
    #reorder the vectors
    for vector in finalOutput:
        rotVec, posVec = quaternionsAndEulers.separateVectorsSimple(vector, usingQuaternions = True)
        rotVec = quaternionsAndEulers.fromQuaternionToEulerVector(rotVec)
        finalOutputOrdered.append(quaternionsAndEulers.glueVectors(rotVec, posVec, vectorHeaderPath = "/home/bee/Desktop/idle animation generator/silenceDataset3secNoHandsCen/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/trn_2023_v0_000_interloctr_silence_0.bvh"))

    with open("resultBvhs/noDifferencesTestEncDecGood.bvh", "w") as f:
        for line in finalOutputOrdered:
            f.write(str(line).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close
'''
#############
# DON'T USE #
#############
#############
# DON'T USE #
#############
def testMultiple(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/multipleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scaler.pkl", "rb"))
    x_input, y, firstPersonPositions, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Validation", verbose=True, specificSize=200, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, scaler = scaler, loadDifferences = False)
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
    print(f"{finalOutput.shape}")
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
    x_input_pos, y, firstPersonPositions_pos, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Train", verbose=True, sequenceSize = n_steps, specificSize=1000, trim=False, scaler=scalerPos, onlyPositions=True, loadDifferences = True, jump=jump)
    x_input_rot, y, firstPersonPositions_rot, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Train", verbose=True, sequenceSize = n_steps, specificSize=1000, trim=False, scaler=scalerRot, onlyRotations=True, loadDifferences = True, jump=jump)
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
    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, partition="Train", datasetName = "silenceDataset3secNoHandsCen", 
                             sequenceSize = n_steps, trim=False, specificSize=10, verbose=True, outSequenceSize=n_steps_out, loadDifferences=True)
    
    print(f"{len(datamodule[0][0][0])}")
    print(f"{datamodule[0][0][0]}")
    print(f"{datamodule[0][0][1]}")
    print(f"{datamodule[0][0][2]}")
    print(f"{datamodule[0][0][3]}")
    print(f"{datamodule[0][0][4]}")
    print(f"{len(datamodule[0][1][0])}")

# method to check if a specific model is created correctly (prints out the summary of a model I'm testing)
def checkNetwork():
    input = keras.Input(shape=(80, 224))
    inputSingle = keras.Input(shape=(1, 224))
    lstm1 = LSTM(1000, activation = 'tanh', return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4)(input)
    lstm2 = LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4)(lstm1)
    dense = Dense(224)(lstm2)
    output = Add()([dense, inputSingle])
    model = keras.Model(inputs=[input, inputSingle], outputs=output, name="test_model")
    model.summary()

def visualizeWeights():
    model = load_model("models/multipleLSTMNoLimbsOneFrame.keras")
    layer = model.layers[0]
    print(f"{layer}")
    weights = layer.get_weights()
    print(f"{weights}")

def prepareScalerForJump(jump, extraInfo, datasetName = "silenceDataset3secNoHandsCen"):
    onlyPos = extraInfo=="onlyPositions"
    onlyRot = extraInfo=="onlyRotations"
    scalerQuaternion = bvhLoader.createAndFitStandardScaler(datasetName = datasetName, jump=jump, onlyPositions=onlyPos, onlyRotations=onlyRot, useQuaternions = True, reorderVectors = True)
    dump(scalerQuaternion, open(f"scalerQuaternionCen{jump}{extraInfo}.pkl", "wb"))
    scalerEuler = bvhLoader.createAndFitStandardScaler(datasetName = datasetName, jump=jump, onlyPositions=onlyPos, onlyRotations=onlyRot, useQuaternions = False, reorderVectors = True)
    dump(scalerEuler, open(f"scalerEulerCen{jump}{extraInfo}.pkl", "wb"))

# def testTheScalerEulerAndQuaternionConversion():
#     scalerEuler = load(open("scalerDifferencesEuler0.pkl", "rb"))
#     scalerQuaternion = load(open("scalerDifferencesQuaternion0.pkl", "rb"))
#     dataPointQuaternionUnscaled, datasetYQuaternion, firstPersonFramesQuaternion, sequencedIdsQuaternion = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Validation", verbose=True, sequenceSize = n_steps, specificSize=10, trim=False, loadDifferences = True, jump=0, useQuaternions = True, reorderVectors = True)
#     dataPointEulerUnscaled, datasetYEuler, firstPersonFramesEuler, sequencedIdsEuler = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3secNoHandsCen", partition="Validation", verbose=True, sequenceSize = n_steps, specificSize=10, trim=False, loadDifferences = True, jump=0, useQuaternions = False, reorderVectors = True)
    
#     printVectors = []
#     for vector in dataPointQuaternionUnscaled[0]:
#         print(vector)
#         vector = scalerQuaternion.inverse_transform(vector.reshape(1, -1))
#         vector = vector.reshape(-1)
#         print(vector)
#         rotVec, posVec = quaternionsAndEulers.separateVectorsSimple(vector, usingQuaternions = True)
#         rotVec = quaternionsAndEulers.fromQuaternionToEulerVector(rotVec)
#         printVectors.append(quaternionsAndEulers.glueVectors(rotVec, posVec, vectorHeaderPath = "/home/bee/Desktop/idle animation generator/silenceDataset3secNoHandsCen/genea2023_trn/genea2023_dataset/trn/interloctr/bvh/trn_2023_v0_000_interloctr_silence_0.bvh"))

#     printVectorsOrdered = []
#     lastVec = firstPersonFramesEuler[0]
#     for diff in dataPointQuaternionUnscaled[0]:
#         printVectorsOrdered.append((lastVec + diff).reshape(-1).tolist())
#         lastVec = (lastVec + diff).tolist().copy()

#     with open("resultBvhs/processTest.bvh", "w") as f:
#         for line in printVectorsOrdered:
#             f.write(str(line).replace("[", "").replace("]", "").replace(",", ""))
#             f.write("\n")
#         # f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
#         f.close    

if __name__ == "__main__":
    # create the standard scaler specifically for the data that we are going to use (just execute once)     
    # prepareScalerForJump(0, "", "silenceDataset3secNoHandsCen")
    # train the model
    train(useDifferences = False, jump=0)
    # test the model
    test("models/DatasetSkipConn0.keras", preparedModel=None)
    # ALL THESE METHODS ARE COMMENTED (I've used them during development to test many things and to make the inference in many ways)
    # testDifferencesAnglesAndPositions("models/differencesNoLimbsOneFramePosJump0.keras","models/differencesNoLimbsOneFrameRotJump0.keras", jump=0)
    # testDifferences("models/differencesDatasetEneko0.keras", jump=0, preparedModel=None)
    # test("models/differencesDataset0.keras")
    # test("models/originalModel.keras")
    # checkNetwork()
    # visualizeWeights()
    # checkDatamodule()