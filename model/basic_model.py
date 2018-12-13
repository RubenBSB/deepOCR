import numpy
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras import backend as K


def convModel(num_classes):
    X0 = Input((1, None, None))

    X = Conv2D(32, (3, 3), input_shape=(1, None, None), activation='relu', padding='same')(X0)
    X = Dropout(0.2)(X)
    X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    x = Dropout(0.2)(X)
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = Dropout(0.2)(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = GlobalAveragePooling2D(data_format='channels_first')(X)
    X = Dropout(0.2)(X)
    X = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(X)
    X = Dropout(0.2)(X)
    X = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(X)
    X = Dropout(0.2)(X)
    X = Dense(num_classes, activation='softmax')(X)

    model = Model(X0, X)
    return model

model = convModel(num_classes)

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

