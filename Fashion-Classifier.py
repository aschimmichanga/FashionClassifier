# Classifying images from Fashion MNIST Kaggle dataset 
# Tutorial: https://www.freecodecamp.org/news/creating-your-first-image-classifier/
# Dataset: https://www.kaggle.com/zalando-research/fashionmnist

# Import statements #################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import  train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Drorpout
from keras.layers import Flatten, BatchNormalization

def main():
    # Preprocessing Data ################################################
    # Import training/testing datasets
    train_df = pd.read_csv("./fashion-mnist_train.csv")
    test_df = pd.read_csv("./fashion-mnist_test.csv")

    # Showing how initial data looks like
    train_df.head()

    # Convert data form from pixel values into numpy array
    train_data = np.array(train_df.iloc[:,1:])
    test_data = np.array(test_df.iloc[:,1:])

    # Get labels by converting categorical data into hot encodings
    train_labels = to_categorical(train_df.iloc[:,0])
    test_labels = to_categorical(test_df.iloc[:,0])

    # Reshape and cast the data into float32
    rows,cols = 28,28

    train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)
    test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    # Normalize data (map pixel values to values between 0 and 1)
    train_data /= 255.0
    test_data /= 255.0


    # Build CNN (Convolutional Neural Network) for modeling image data ##
    train_x, val_x, train_y, val_y = train_test_split(train_data, 
                                                      train_labels, 
                                                      test_size=0.2)

    batch_size = 256
    epochs = 50
    input_shape = (rows, cols, 1)

    model = baseline_model()
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(val_x, val_y))
    predictions = model.predict(test_data)

def baseline_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation= "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model
    