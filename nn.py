from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

class NN:
    def __init__(self):
        self.input_shape = ''   
        self.classes = ''     

    def LeNet(self, input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same",
                        input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=5, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # Output Layer
        if classes==1:
            model.add(Dense(classes))
        else:
            model.add(Dense(classes))
            model.add(Activation("softmax"))  
        return model
        
    def AlexNet(self, input_shape, classes):
        model = Sequential()
        #第一段  
        model.add(Conv2D(filters=96, kernel_size=(11,11),
                         strides=(4,4), padding='same',                 
                         input_shape=input_shape,                 
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
        #第二段 
        model.add(Conv2D(filters=256, kernel_size=(5,5),
                         strides=(1,1), padding='same',                                  
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
        #第三段
        model.add(Conv2D(filters=384, kernel_size=(3,3), 
                        strides=(1,1), padding='same', 
                        activation='relu'))
        model.add(Conv2D(filters=384, kernel_size=(3,3), 
                        strides=(1,1), padding='same', 
                        activation='relu')) 
        model.add(Conv2D(filters=256, kernel_size=(3,3), 
                        strides=(1,1), padding='same', 
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                                strides=(2,2), padding='same'))
        #第四段
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        # Output Layer
        if classes==1:
            model.add(Dense(classes))
        else:
            model.add(Dense(classes))
            model.add(Activation("softmax"))            
        return model
