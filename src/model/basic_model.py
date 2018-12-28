import numpy
from data.data_preprocess import get_dataset, get_vocab
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras import backend as K

X,words = get_dataset("../data")
X = np.reshape()

word_to_index, index_to_word = get_vocab(words)
num_classes = len(word_to_index)
Y = [word_to_index[word] for word in words]
Y = to_categorical(Y,num_classes)

def convModel(num_classes):
    X0 = Input((200, 200, 1))

    X = Conv2D(32, (3, 3), input_shape=(200, 200, 1), activation='relu')(X0)
    X = Dropout(0.2)(X)
    X = Conv2D(32, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    x = Dropout(0.2)(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = GlobalAveragePooling2D()(X)
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
lr = 0.01
epsilon = 10**(-8)
decay = lr/epochs
opt = Adam(lr=lr, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

model.fit(X,Y)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

