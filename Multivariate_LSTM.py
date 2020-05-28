# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:37:02 2020
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/@author: hojat behrooz
"""
from pandas import DataFrame , concat
from sklearn.preprocessing import LabelEncoder , MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot

from pandas import read_csv
from matplotlib import pyplot
from pandas import read_csv
from datetime import datetime
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)  

from statsmodels.tsa.vector_ar.var_model import VAR 
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from  input1 import input_data , input_single ,average_corr ,read_weights,\
                    delete_zeros , moving_average,Umoving_average,corr_matrix,\
                    input_consider,dif_m,wm,mean_t,ma_m,Uma_m,dif, dif_m,\
                    integ_m,integ_level,logn_f,exp_f,normalize,adf_test,make_weights   
datatype=['US_DEATH_CASES',  
'US_CONFIRMED_CASES',
'GLOBAL_DEATH_CASES',
'GLOBAL_CONFRIRMED_CASES']
US_DEATH=0  
US_CONFIRMED =1  
GLOBAL_DEATH=2
GLOBAL_CONFRIRMED=3
np.seterr(divide='ignore')

                   
      
epi= 0.0001 # precious data                            
Alpha=.05 

# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset_type,flatten,name,Y,last_date,first_date =input_data()

# making a data frame of input data with date index and column name of regions
#delete dataset with finall value less than 1/10,000th total cases
valuable=(np.array(Y)[:,-1:]>epi*np.sum(np.array(Y)[:,-1:])).flatten()
most_valuable=np.array(sorted(zip(np.array(Y)[:,-1:].flatten(),name),reverse=True))
y=[Y[i] for i in range(len(Y)) if valuable[i]]
namep=[name[i] for i in range(len(name)) if valuable[i]]
#weights=read_weights(namep,dataset_type) 
weights=make_weights(namep,dataset_type)
difrant=3
d=np.array(y,dtype='float64')
d=d.transpose()
d= dif_m(d)

# # ensure all data is float
values = d.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
n_in,n_out=27,7
reframed = series_to_supervised(scaled, n_in, n_out)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_test = n_out
n_region=scaled.shape[1]
train = values[:-n_test, :]
test = values[-n_test:, :]
# split into input and outputs
train_X, train_y = train[:, :-n_region], train[:, -n_region:]
test_X, test_y = test[:, :-n_region], test[:, -n_region:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

################# design network fo simple LSTM
# model = Sequential()
# #return_sequences: Boolean. Whether to return the last output
# #in the output sequence, or the full sequence.return_sequences=True,
# model.add(LSTM(68,activation='relu',  \
#                 input_shape=(train_X.shape[1], train_X.shape[2])))
# #model.add(LSTM(100, activation='relu'))

# #model.add(LSTM(7, activation='relu'))
# model.add(Dense(n_region))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=150, batch_size=48, 
#             validation_data=(test_X, test_y), verbose=2, shuffle=False)

###################################################
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
model = Sequential()
model.add(LSTM(256, activation='relu', \
                input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(RepeatVector(n_out))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(n_region))
model.compile(optimizer='adam', loss='mse')
# fit model

X,y = train_X,train_y
#y = y.reshape((y.shape[0], 1, y.shape[1]))
rtest_y=test_y.reshape(test_y.shape[0],1,test_y.shape[1])
history=model.fit(X, y, epochs=100, verbose=2,validation_data=(test_X, test_y), shuffle=False)



# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction

yhat = model.predict(test_X)
#yhat=yhat.reshape(yhat.shape[0],yhat.shape[2])
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(yhat)
#inv_yhat = inv_yhat[:,0]
# invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(test_y)
#inv_y = inv_y[:,0]
# calculate RMSE
#inv_yhat=integ_level(inv_yhat,difrant-1)
#inv_y=integ_level(inv_y,difrant-1)


WMAE=0
for j in range(inv_y.shape[1]):
  WMAE+=mean_absolute_error(inv_y[:,j],inv_yhat[:,j])*weights[j]
  
rmse = mean_absolute_error(inv_y, inv_yhat)
print('Test RMSE: %.3f WMAE: %.2f' % (rmse,WMAE))
if(dataset_type in [US_DEATH,GLOBAL_DEATH]):print("Fatalities score =",10*WMAE/(2*len(namep)))
else: print("score =",WMAE/(2*len(namep)))