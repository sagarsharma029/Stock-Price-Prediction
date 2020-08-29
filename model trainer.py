from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json
import numpy as np
import math
import pandas as pd

#df = pd.read_csv('TATA.csv')
#df = df.iloc[::-1].reset_index(drop=True)

#df = pd.read_csv('GOOGL.csv')

df = pd.read_csv('TSLA.csv')

#spitting data for training
data = df.filter(['Close'])
data_set = data.values
train_len = math.ceil(len(data)*0.8)

#scailing data for training
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

#scailing training data
train_data = scaled_data[0:train_len, : ]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#creating neural network(LSTM)
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#training the LSTM RNN
model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
#epochs are number of times you want to train your network.
#more epochs means more training time but higher acuuracy(mostly).
model.fit(x_train, y_train,batch_size = 1, epochs = 10)

#saving model and weights for future use
model_json = model.to_json()
with open('model.json',"w") as json_file:
    json_file.write(model_json)

model.save('model-weights.hdf5')
