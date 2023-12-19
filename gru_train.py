import matplotlib.pyplot as plt
import pandas_datareader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
import datetime as dt

# from pandas_datareader import data as pdr
plt.style.use('fivethirtyeight')

import yfinance as yf

yf.pdr_override()

# get the stock quote

start = dt.datetime(2010, 3, 26)
end = dt.datetime(2023, 12, 18)
data = pandas_datareader.data.get_data_yahoo('AAPL', start=start, end=end)

# creating a new dataframe with only the 'Close' column
df = data.filter(["Close"])
# converting data frame into a numpy array
dataset = df.values
# getting the number of rows to train the model on

training_data_len = math.ceil(
    len(dataset) * 0.8)  # since it has to be trained about 80% thus I have multiplied it with .8
# math.ceil used to round up the data
# training_data_len

# scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))  # scaling the values to be in between 0 and 1
scaled_data = scaler.fit_transform(
    dataset)  # now transforming the data to be in range 0 to 1. it contains the dataset that has now been scaled
# from sklearn.externals import joblib
# import sklearn.external.joblib as extjoblib

import joblib
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)
# create the training dataset
# create the scaled training dataset
trained_data = scaled_data[0:training_data_len,
               :]  # This will contain all of the data from 0 to transformed data with the coulmns
# split the data into x_train and y_train data sets
x_train = []  # Independent training variable
y_train = []  # Dependent training variable

for i in range(100, len(trained_data)):
    x_train.append(trained_data[i - 100:i,
                   0])  # appending the past 60 values to x_train; from position i-60 to i without including i
    y_train.append(trained_data[i, 0])  # contain the 60 first values. 0 is the position column
    # if i<=101:
    #    print(x_train)
    #    print(y_train)

# Converting the x_train and y_train to numpy arrays for training the LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

# # Reshaping  the data because LSTM model requires 3D value
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape


import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import RootMeanSquaredError

model = tf.keras.Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(100, 1)))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2))

# GRU Layers
model.add(Bidirectional(GRU(100, return_sequences=False)))
model.add(Dropout(0.5))

# Final Layers
model.add(Dense(1, activation='linear'))

# compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# the optimizer is used to improve upon the loss function and the loss function is used to measure how well the model did on training

# train the model
model.fit(x_train, y_train, batch_size=50, epochs=5)  # fit is another name for train. epochs is number of iterations

# create the testing data set
# create a new array containing scaled values from index 2664 to 3404
test_data = scaled_data[training_data_len - 100:, :]  # This contains the data from the 2664 to 3404
# create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(100, len(test_data)):
    x_test.append(test_data[i - 100:i, 0])  # appending the past 100 values to test dataset

# convert the data to a numpy array
x_test = np.array(x_test)

# convert the data to a numpy array
x_test = np.array(x_test)

# Reshaping of data
x_test = np.reshape(x_test,
                    (x_test.shape[0], x_test.shape[1], 1))  # 1 is the number of features which is the close price

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(
    predictions)  # unscaling the values. This will contain the values that the y_test dataset contains

# get the quote
apple_quote = pandas_datareader.data.get_data_yahoo('AAPL', start=start, end=end)
#  new dataframe
new_df = apple_quote.filter(['Close'])
# getting the last 60 days closing price values and cinverting the dataframe to an array
last_100_days = new_df[-100:].values
# sclaing the data to be the values between 0 and 1
last_100_days_scaled = scaler.transform(last_100_days)
# create an empty list
X_test = []
# append the past 60 days
X_test.append(last_100_days_scaled)
# converting the X_test data set to a numpy array
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# getting the predicted scaled price
pred_price = model.predict(X_test)
# undo the sclaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
model.save('gru_model')
