import tensorflow as tf
from tensorflow import keras

def alexnet(in_shape=[227,227,3],n_classes=1000,opt='sgd'):
    input_layer = keras.layers.Input(shape=in_shape,name='In')
    model = keras.layers.Conv2D(filters=96,kernel_size=(11,11),strides=4,padding="valid",activation="relu",name='C1')(input_layer)
    model = keras.layers.MaxPool2D(pool_size=(3, 3), padding='valid',strides=2,name='S2')(model)
    model = keras.layers.Conv2D(filters=256,kernel_size=(5,5),strides=1,padding="same",activation="relu",name='C3')(model)
    model = keras.layers.MaxPool2D(pool_size=(3, 3), padding='valid',strides=2,name='S4')(model)
    model = keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding="same",activation="relu",name='C5')(model)
    model = keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding="same",activation="relu",name='C6')(model)
    model = keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding="same",activation="relu",name='C7')(model)
    model = keras.layers.MaxPool2D(pool_size=(3, 3), padding='valid',strides=2,name='S8')(model)
    model = keras.layers.Flatten()(model)
    model = keras.layers.Dense(units=4096,activation='relu',name='F9')(model)
    model = keras.layers.Dense(units=4096,activation='relu',name='F10')(model)
    output_layer = keras.layers.Dense(units=n_classes,activation='softmax',name='Out')(model)
    
    model = keras.Model(input_layer,output_layer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model