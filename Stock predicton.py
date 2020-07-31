from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

'''df = pd.read_csv('TATA.csv')
df = df.iloc[::-1].reset_index(drop=True)'''

#df = pd.read_csv('GOOGL.csv')

df = pd.read_csv('TSLA.csv')

"""
chart to show history of stock price-
plt.figure(figsize=(16,8))
plt.title('Closing price')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price (₹)', fontsize = 18)
plt.plot(df['Close'])
"""

data = df.filter(['Close'])
data_set = data.values
train_len = math.ceil(len(data)*0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

train_data = scaled_data[0:train_len, : ]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

model.fit(x_train, y_train,batch_size = 1, epochs = 10)

test_data = scaled_data[train_len-60: , : ]
x_test = []
y_test = data_set[train_len: , : ]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

train = data[ :train_len]
valid = data[train_len:]
valid['Predictions'] = predictions
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 20)
plt.ylabel('Closing Price (₹)', fontsize = 20)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()
