import matplotlib.pyplot as plt
import pandas_datareader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import datetime as dt
from datetime import datetime, timedelta

# from pandas_datareader import data as pdr
plt.style.use('fivethirtyeight')

import yfinance as yf

yf.pdr_override()

# get the stock quote
#
# start = dt.datetime(2023, 10, 22)
# end = dt.datetime(2023, 12, 22)


# prompt: write and inferance function for the model trained
import tensorflow as tf
import keras
from tensorflow.keras.regularizers import L1, L2


# def predict_():
#   end = datetime.now()
#   days_before = 100
#   start = end - timedelta(days=days_before)
#   loaded_model = tf.keras.models.load_model('gru_model')
#   # get the quote
#   apple_quote = pandas_datareader.data.get_data_yahoo('AAPL',start= start, end= end)
#   #  new dataframe
#   new_df= apple_quote.filter(['Close'])
#   # getting the last 60 days closing price values and cinverting the dataframe to an array
#   last_100_days = new_df[-100:].values
#   # sclaing the data to be the values between 0 and 1
#   # scaling the data
#   scaler = MinMaxScaler(feature_range=(0, 1))  # scaling the values to be in between 0 and 1
#   # scaled_data = scaler.fit_transform(dataset)  # now transforming the data to be in range 0 to 1. it contains the dataset that has now been scaled
#   last_100_days_scaled = scaler.transform(last_100_days)
#   # create an empty list
#   X_test =[]
#   # append the past 60 days
#   X_test.append(last_100_days_scaled)
#   # converting the X_test data set to a numpy array
#   X_test = np.array(X_test)
#   # reshape the data
#   X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#   # getting the predicted scaled price
#   print("(X_test.shape",X_test.shape)
#   pred_price= loaded_model.predict(X_test)
#   # undo the sclaling
#   pred_price =  scaler.inverse_transform(pred_price)
#   print(pred_price)
#   pred_price
#
#

def predict_(end,ticker='AAPL'):
    # end = datetime.now()
    days_before = 100
    start = end - timedelta(days=days_before)
    loaded_model = tf.keras.models.load_model('gru_model')
    # get the quote
    apple_quote = pandas_datareader.data.get_data_yahoo('AAPL', start=start, end=end)
    #  new dataframe
    new_df = apple_quote.filter(['Close'])
    # getting the last 60 days closing price values and cinverting the dataframe to an array
    last_100_days = new_df[-100:].values
    # scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))  # scaling the values to be in between 0 and 1
    # sclaing the data to be the values between 0 and 1
    import joblib
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)  # .dump(scaler, scaler_filename)
    last_100_days_scaled = scaler.transform(last_100_days)
    print(len(last_100_days_scaled))
    # create an empty list
    X_test = []
    # append the past 60 days
    X_test.append(last_100_days_scaled)
    # converting the X_test data set to a numpy array
    print(len(X_test[0]))
    X_test = np.array(X_test)
    # reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    from keras.preprocessing.sequence import pad_sequences

    # Assuming your original sequences are stored in 'sequences'
    X_test = pad_sequences(X_test, maxlen=100, padding='post', truncating='post')

    print(X_test.shape)
    # getting the predicted scaled price
    pred_price = loaded_model.predict(X_test)
    # undo the sclaling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0])
    return pred_price[0][0]


# predict_()
