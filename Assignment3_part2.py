"""
Created on Mon Aug 10 00:06:25 2020

@author: AISHWARYA
"""

#import sys
from keras.models import Sequential
#from keras.models import Dense
#from keras.models import Conv2D
#from keras.models import MaxPooling2D
#from keras.models import Flatten
import keras
from keras.models import load_model
#from keras.layers import Dense
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers.convolutional import Conv2D

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

#from keras.layers import Convolution2D as Conv2D
#from keras.layers.convolutional import Deconv2D as Conv2DTranspose
#from tensorflow.python.keras.datasets import mnist
#from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.utils import np_utils
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os

from skimage.transform import resize

model=Sequential()

model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(output_dim=128,activation='relu',init='random_uniform'))
#model.add(Dense(output_dim=1,activation='sigmoid',init='random_uniform'))
model.add(Dense(output_dim=3,activation='sigmoid',init='random_uniform'))

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train = train_datagen.flow_from_directory(r'C:\\Users\\AISHWARYA\\Desktop\\dataset\\trainset', target_size=(64,64),batch_size=32,class_mode='categorical')
x_test = train_datagen.flow_from_directory(r'C:\\Users\\AISHWARYA\\Desktop\\dataset\\testset', target_size=(64,64),batch_size=32,class_mode='categorical')

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(x_train,samples_per_epoch=8000,epochs=25,validation_data=x_test,nb_val_samples=2000)
 
model.save('flowerModel.h5')


model = load_model('flowerModel.h5')
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

def detect(frame):
    try:
        img = resize(frame,(64,64))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img=img/255.0
        prediction =model.predict(img)
        print(prediction)
        prediction_class=model.predict_classes(img)
        print(prediction_class)
    except AttributeError:
        print ("shape not found")
   #file which needs to be predicted     
frame= cv2.imread("C:\\Users\\AISHWARYA\\Desktop\\cat.jpg")
data=detect(frame)

