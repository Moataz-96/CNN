import tensorflow as tf
from tensorflow import keras

def vgg16(in_shape=[224,224,3],n_classes=1000,opt='sgd'):
    conv_vgg = partial(keras.layers.Conv2D,kernel_size=3,strides=1, activation='relu', padding="SAME")
    pool_vgg = partial(keras.layers.MaxPool2D,pool_size=(2,2),strides=None)
    
    model = keras.models.Sequential([
        keras.layers.Input(shape=in_shape,name='In'),
        
        conv_vgg(filters=64),
        conv_vgg(filters=64),
        pool_vgg(),
        
        conv_vgg(filters=128),
        conv_vgg(filters=128),
        pool_vgg(),
        
        conv_vgg(filters=256),
        conv_vgg(filters=256),
        conv_vgg(filters=256),
        pool_vgg(),
        
        conv_vgg(filters=512),
        conv_vgg(filters=512),
        conv_vgg(filters=512),
        pool_vgg(),
        
        conv_vgg(filters=512),
        conv_vgg(filters=512),
        conv_vgg(filters=512),
        pool_vgg(),
        
        keras.layers.Flatten(),
        keras.layers.Dense(units=4096,activation='relu'),
        keras.layers.Dense(units=4096,activation='relu'),
        keras.layers.Dense(units=n_classes,activation='softmax')
         
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model