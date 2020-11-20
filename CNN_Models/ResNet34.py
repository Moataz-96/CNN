import tensorflow as tf
from tensorflow import keras

class ResUnit(keras.layers.Layer):
    def __init__(self,name,filters=64,strides=1, **kwargs):
        super().__init__(**kwargs)
        self.layer_name=name
        self.strides =strides
        self.filters = filters
        self.activation1 = keras.layers.Activation('relu',name=str(self.layer_name[0])+'_relu')
        self.conv1 = keras.layers.Conv2D(filters=self.filters,kernel_size=(3,3),strides=self.strides,padding="SAME",name=str(self.layer_name[0])+"_main_"+self.layer_name[1])
        self.bn1 = keras.layers.BatchNormalization(name=str(self.layer_name[0])+'_batch_norm')
        
        self.activation2 = keras.layers.Activation('relu',name=str(self.layer_name[0]+1)+'_relu')
        self.conv2 = keras.layers.Conv2D(filters=self.filters,kernel_size=(3,3),strides=(1,1),padding="SAME",name=str(self.layer_name[0]+1)+"_main_"+self.layer_name[1])
        self.bn2 = keras.layers.BatchNormalization(name=str(self.layer_name[0]+1)+'_batch_norm')
        
        
        
    def __call__(self,inputs):
        if(self.strides > 1):
            skip_path = keras.layers.Conv2D(filters=self.filters,kernel_size=(1,1),strides=self.strides,padding="SAME",name=str(self.layer_name[0]+1)+'_skip_conv_1x1')(inputs)
            skip_path = keras.layers.BatchNormalization(name=str(self.layer_name[0]+1)+'_skip_batch_norm')(skip_path)
        else:
            skip_path = inputs
            
        main_path = self.conv1(inputs)
        main_path = self.bn1(main_path)
        main_path = self.activation1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.bn2(main_path)
        
        model = keras.layers.Add(name=str(self.layer_name[0]+1)+'_add')([skip_path,main_path])
        return self.activation2(model)

def ResNet34(in_shape = [224,224,3],n_classes=1000,opt='sgd'):
    in_layer = keras.layers.Input(shape=in_shape,name='input')
    model = keras.layers.Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding="SAME",name='1_7x7_conv_64_2')(in_layer)
    model = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding="SAME",name='1_pool/2')(model)
    
    prev_filter = 64
    idx_layer = 2
    for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
        strides = 1 if prev_filter == filters else 2
        name = "3x3_conv_" + str(filters) + "_"+ str(strides) 
        model = ResUnit(name=[idx_layer,name],filters=filters,strides=strides)(model)
        idx_layer += 2
        prev_filter = filters
    
    model = keras.layers.GlobalAveragePooling2D(name='34_global_avg_pool_')(model) ##Same For Flatten()
    out_layer = keras.layers.Dense(units=n_classes,activation='softmax',name='softmax_output')(model)
    
    model = keras.Model(inputs=in_layer, outputs=out_layer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model