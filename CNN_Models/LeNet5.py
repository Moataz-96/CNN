import tensorflow as tf
from tensorflow import keras

def lenet_5(in_shape=[28, 28, 1], n_classes=10, opt='sgd'):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=in_shape,name='orig_in'))
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2),name='In'))
    model.add(keras.layers.Conv2D(filters=6,kernel_size=(5,5),strides=1,padding="valid",activation="tanh",name='C1'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid',name='S2'))
    model.add(keras.layers.Conv2D(filters=16,kernel_size=(5,5),strides=1,padding="valid",activation="tanh",name='C3'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid',name='S4'))
    model.add(keras.layers.Conv2D(filters=120,kernel_size=(5,5),strides=1,padding="valid",activation="tanh",name='C5'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=84,activation='tanh',name='F6'))
    model.add(keras.layers.Dense(units=n_classes,activation='softmax',name='Out'))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model