from keras import models
from keras import layers
from keras.optimizers import Adam
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import random
import tensorflow as tf
from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])

f_num = 150

with tf.device("/cpu:0"):
    dataset = numpy.loadtxt('Gaussian_Distance_' + str(f_num) + '_img_features_43.81.csv', delimiter=",", skiprows=1)
    random.shuffle(dataset)

    train_index = len(dataset) * 0.7
    train_index = int(train_index)

    train_data = dataset[:train_index, :-1]
    train_labels = dataset[:train_index, -1]

    test_data = dataset[train_index:, :-1]
    test_labels = dataset[train_index:, -1]

    train_labels = to_categorical(train_labels, 4)
    test_labels = to_categorical(test_labels, 4)

    network = models.Sequential()
    network.add(layers.Dense(50, activation='relu', input_dim=f_num))
    network.add(layers.Dense(10, activation='relu'))
    network.add(layers.Dense(4, activation='softmax'))

    network.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='acc', patience=5, mode='auto')
    network.fit(train_data, train_labels, epochs=100)

    test_loss, test_acc = network.evaluate(test_data, test_labels)
    test_acc = round(test_acc, 3)

    print('test_acc :', test_acc)

    K.clear_session()
