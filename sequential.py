import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(784,)), 
    keras.layers.Dense(128, activation='relu'),  
    keras.layers.Dense(10, activation='softmax')  

model.compile(optimizer='adam',  
              loss='categorical_crossentropy', 
              metrics=['accuracy'])  

model.summary()
