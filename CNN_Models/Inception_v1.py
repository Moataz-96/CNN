import tensorflow as tf
from tensorflow import keras

class InceptionModule(keras.layers.Layer):
    def __init__(self,filters,name, strides=1, activation="relu", **kwargs):
        #filters = [c1,c1_3,c1_5,c_3,c_5,c_pool] 
        
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.filters = filters
        self.layer_name = name
        self.conv1 = keras.layers.Conv2D(filters=self.filters[0],kernel_size=(1,1),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_1x1')
        
        self.conv1_3 = keras.layers.Conv2D(filters=self.filters[1],kernel_size=(1,1),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_3x3_reduced')
        self.conv1_5 = keras.layers.Conv2D(filters=self.filters[2],kernel_size=(1,1),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_5x5_reduced')
        
        self.conv3 = keras.layers.Conv2D(filters=self.filters[3],kernel_size=(3,3),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_3x3')
        self.conv5 = keras.layers.Conv2D(filters=self.filters[4],kernel_size=(5,5),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_5x5')
        
        self.conv_pool = keras.layers.Conv2D(filters=self.filters[5],kernel_size=(1,1),strides=strides,padding="SAME",activation=self.activation,name=self.layer_name+'_conv_1x1_pool_proj')
        
        self.maxpool = keras.layers.MaxPool2D(pool_size=(3,3),strides=strides,padding="SAME",name=self.layer_name+'_max_pool_3x3')
        
    
    def __call__(self,inputs):
        conv_filter1 = self.conv1(inputs)
        
        conv_filter3 = self.conv1_3(inputs)
        conv_filter3 = self.conv3(conv_filter3)
        
        conv_filter5 = self.conv1_5(inputs)
        conv_filter5 = self.conv5(conv_filter5)
        
        
        conv_filters_pool = self.maxpool(inputs)
        conv_filters_pool = self.conv_pool(conv_filters_pool)
        
        
        conv_concat = keras.layers.concatenate([conv_filter1,conv_filter3,conv_filter5,conv_filters_pool],name=self.layer_name,axis=-1)
        
        return conv_concat   

class InceptionAuxiliary(keras.layers.Layer):
    def __init__(self,name,filters=128,fc1=1024,num_classes=1000,strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.layer_name=name
        self.activation = keras.activations.get(activation)
        self.avgpool = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3,name=self.layer_name+'_avg_pooling_5x5/3')
        self.conv = keras.layers.Conv2D(filters=filters,kernel_size=(1,1),strides=(1,1),padding="SAME",activation=self.activation,name=self.layer_name+'_conv_1x1')
        self.fc1 = keras.layers.Dense(units = fc1,activation=self.activation,name=self.layer_name+'_fc_1024')
        self.out = keras.layers.Dense(units=num_classes,activation='softmax',name=self.layer_name+'_output')
        
    def __call__(self,inputs):
        model = self.avgpool(inputs)
        model = self.conv(model)
        model = keras.layers.Flatten()(model)
        model = self.fc1(model)
        model = keras.layers.Dropout(rate=0.7)(model)
        model = self.out(model)
        return model

#Use MaxPool(padding = same) or use ZeroPadding before MaxPooling

def Inception(in_shape = [224,224,3],n_classes=1000,opt='sgd'):
    in_layer = keras.layers.Input(shape=in_shape,name='Input')
    model = keras.layers.Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding="SAME",activation='relu',name='conv_1_7x7/2')(in_layer)
    
    model = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding="SAME",name='max_pool_1_3x3/2')(model)
    
    model = keras.layers.Conv2D(filters=64,kernel_size=(1,1),strides=(1,1),padding="SAME",activation='relu',name='conv_2_3x3/2_reduced')(model)
    model = keras.layers.Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),padding="SAME",activation='relu',name='conv_2_3x3/2')(model)
    
    model = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding="SAME",name='max_pool_2_3x3/2')(model)
    
    model = InceptionModule([64,96,16,128,32,32],name='3a')(model)
    model = InceptionModule([128,128,32,192,96,64],name='3b')(model)

    model = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding="SAME",name='max_pool_3_3x3/2')(model)
    
    model_fork = InceptionModule([192,96,16,208,48,64],name='4a')(model)
    
    model = InceptionModule([160,112,24,224,64,64],name='4b')(model_fork)
    model_aux1 = InceptionAuxiliary(num_classes=n_classes,name='auxiliary1')(model_fork)
    
    model = InceptionModule([128,128,24,256,64,64],name='4c')(model)
    model_fork = InceptionModule([112,144,32,288,64,64],name='4d')(model)
    
    model = InceptionModule([256,160,32,320,128,128],name='4e')(model_fork)
    model_aux2 = InceptionAuxiliary(num_classes=n_classes,name='auxiliary2')(model_fork)
    
    model = keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding="SAME",name='max_pool_4_3x3/2')(model)
    
    model = InceptionModule([256,160,32,320,128,128],name='5a')(model)
    model = InceptionModule([384,192,84,384,128,128],name='5b')(model)
    
    model = keras.layers.GlobalAveragePooling2D(name='global_avg_pool_5_7x7/1')(model) ##Same For Flatten()
    
    model = keras.layers.Dropout(rate=0.4,name='dropout_output')(model)
    out_layer = keras.layers.Dense(units=n_classes,activation='softmax',name='softmax_output')(model)
    
    model = keras.Model(inputs=in_layer, outputs=[out_layer,model_aux1,model_aux2])
    
    #The total loss used by the inception net during training.
    #total_loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2
    
    model.compile(loss=["sparse_categorical_crossentropy"
                       ,"sparse_categorical_crossentropy"
                       ,"sparse_categorical_crossentropy"]
                       ,loss_weights=[0.4,0.3,0.3]
                       , optimizer=opt, metrics=["accuracy"])
    return model    