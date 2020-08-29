from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

#df = pd.read_csv('TATA.csv')
#df = df.iloc[::-1].reset_index(drop=True)

#df = pd.read_csv('GOOGL.csv')

df = pd.read_csv('TSLA.csv')

#splitting data
data = df.filter(['Close'])
data_set = data.values
train_len = math.ceil(len(data)*0.8)

#scaling data for model
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_set)

test_data = scaled_data[train_len-60: , : ]
x_test = []
y_test = data_set[train_len: , : ]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#reshaping data
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#loading saved (pre-trained) model and weights
model = model_from_json(open("model.json", "r").read())
model.load_weights('model-weights.hdf5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

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
