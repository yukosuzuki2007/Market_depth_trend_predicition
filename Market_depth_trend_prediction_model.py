# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:38:29 2020

@author: alial
"""

# imports
import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

# The following lines are to setup my GPU for the learning
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
# End of GPU setup


db = sqlite3.connect(r'C:\Udemy\Interactive Brokers Python API\streaming ES\ES_ticks1.db') # Loading the file
df = pd.read_sql_query("SELECT * from ES_market_depth", db) # converting file into a Dataframe
df['difference'] = df['lastPrice'].shift(1)-df['lastPrice'] # setting the difference between two tics (current and last tick) and shifting one row up 
df.dropna(inplace = True) 

# setting the direction of the trend and separting into categories for learning

"""
same with a value of 1 means the trend is not changing
uptrend with a value of 1 means the trend is up
downtrend with a value of 1 means the trend is down
"""

df['direction'] = df['difference'].apply(lambda x: 1 if x>0 else -1 if x<0 else 0) 
df['same'] = 1
df['up_trend'] =0
df['down_trend'] = 0
for i in range(len(df)):
    if df['direction'].iloc[i] == 0:
        df.iloc[i,-3] = 1
        df.iloc[i,-2] =0
        df.iloc[i,-1] = 0
    elif df['direction'].iloc[i] == -1:
        df.iloc[i,-3] = 0
        df.iloc[i,-2] =0
        df.iloc[i,-1] = 1
    else:
        df.iloc[i,-3] = 0
        df.iloc[i,-2] = 1
        df.iloc[i,-1] = 0


# balancing data to contain same directional data to help learning without biases
df = pd.concat(objs = (pd.DataFrame(df[df['same']>0].iloc[0:len(df[df['down_trend']>0])]), pd.DataFrame(df[df['up_trend']>0].iloc[0:len(df[df['down_trend']>0])]), pd.DataFrame(df[df['down_trend']>0].iloc[0:len(df[df['down_trend']>0])]))).sort_index()

#Setting X and y
X = df.iloc[:,1:-6].values
y = df.iloc[:,-3:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Normalizing data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#setting up the model of tensorflow
input_layer = Input(shape=(X.shape[1]))
x = Dense(50, activation='relu')(input_layer)
x = Dropout(0.5)(x)
x = Dense(25, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(y.shape[1], activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

#creating an early stop based on minmizing val_loss
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

#fit the model
r = model.fit(X_train, y_train, epochs = 200000,
             validation_data = (X_test, y_test), callbacks=[early_stop], shuffle=0.5)

#plot the results.
pd.DataFrame(r.history).plot()

