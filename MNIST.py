# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.datasets import mnist
from keras.utils import np_utils
def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train.reshape(x_train.shape[0], 28*28)
        x_test = x_test.reshape(x_test.shape[0], 28*28)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train/255
        x_test = x_test/255
    
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
(x_train,y_train),(x_test,y_test)=load_data()

#%%
from keras.models import Sequential
from keras.layers.core import Dense
def build_model():
        
        model = Sequential()
        
        model.add(Dense(input_dim=28*28,
                  units=500,activation='relu'))
        model.add(Dense(units=500,activation='relu'))
        model.add(Dense(units=500,activation='relu'))
        model.add(Dense(units=10,activation='softmax'))
        model.summary()
        return model
    
model = build_model()

#%%
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)
score = model.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])
score = model.evaluate(x_test,y_test)
print ('\nTest Acc:', score[1])