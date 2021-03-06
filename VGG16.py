from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import os
from keras import models
from keras.applications import *
from keras import layers, optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed
from tensorflow.keras import optimizers
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

base_dir = 'C:/Users/Gachon/Desktop/exp/Experiment_original_mel_60'
train_dir = base_dir + '/' + 'train'
test_dir = base_dir + '/' + 'test'
val_dir = base_dir + '/' + 'validation'

# local_device_protos = device_lib.list_local_devices()
# print([x.name for x in local_device_protos if x.device_type == 'GPU'])

train_data = []
train_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(train_dir + '/' + category)
    if category == 'normal':
        data_list = data_list[:1117]
    for data_name in data_list:
        img = image.load_img(train_dir + '/' + category + '/' + data_name, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        train_data.append(img_tensor)
        train_label.append(label)

val_data = []
val_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(val_dir + '/' + category)
    for data_name in data_list:
        img = image.load_img(val_dir + '/' + category + '/' + data_name, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        val_data.append(img_tensor)
        val_label.append(label)

test_data = []
test_label = []
for label, category in enumerate(["crackle", "wheeze", "normal", "mixed"]):
    data_list = os.listdir(test_dir + '/' + category)
    for data_name in data_list:
        img = image.load_img(test_dir + '/' + category + '/' + data_name, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        test_data.append(img_tensor)
        test_label.append(label)

train_data = [train_data]
test_data = [test_data]
val_data = [val_data]

train_label = to_categorical(train_label, 4)
test_label = to_categorical(test_label, 4)
val_label = to_categorical(val_label, 4)

while True:
    # with tf.device("/gpu:0"):

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten(name='flatten'))
    # print(model.summary())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='auto')
    history = model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=60, batch_size=10,
                        callbacks=[early_stopping])

    """
    predictions = model.predict_generator(test_data, steps=np.math.ceil(test_data.samples / test_data.batch_size),)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data.classes
    class_labels = list(test_data.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    accuracy = accuracy_score(true_classes, predicted_classes)
    """

    test_loss, test_acc = model.evaluate(test_data, test_label, batch_size=10)
    acc = round(test_acc * 100, 2)
    print("===============================================================================================================================")
    print('Accuracy : ' + str(acc))
    print("===============================================================================================================================")
    '''
    f = open("VGG16_result.txt", 'w')
    f.write("acc: " + acc)
    f.close()
    '''
    model.save('save_model_' + str(acc) + '.h5')

    K.clear_session()