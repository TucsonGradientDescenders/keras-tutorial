from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import mean_squared_error
from keras import optimizers

import tarfile
import os
import numpy as np

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()
print(testY[0])

num_features = trainX.shape[1]
num_labels = trainY.shape[1]

model = Sequential()
#model.add(Dense(int(num_features/10), input_dim=num_features, activation='relu'))
model.add(Dense(num_labels, input_dim=num_features, activation='softmax'))
num_epochs = 3000

sgd = optimizers.SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainX, trainY, batch_size=128, nb_epoch=num_epochs,
	verbose=1, validation_split=0.3)

score = model.evaluate(testX, testY, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])