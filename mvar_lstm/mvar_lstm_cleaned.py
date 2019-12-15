# Adapted from https://www.kaggle.com/sebask/keras-2-0

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gc

# Viz
import matplotlib.pyplot as plt

# importing libraries required for our model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint

# Import data
sales = pd.read_csv('data/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('data/shops.csv')
items = pd.read_csv('data/items.csv')
cats = pd.read_csv('data/item_categories.csv')
val = pd.read_csv('data/test.csv')

# Rearrange the raw data to be monthly sales by item-shop
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.,inplace=True)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
df.head()

# Merge data from monthly sales to specific item-shops in test data
test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

# Strip categorical data so keras only sees raw timeseries
test = test.drop(labels=['ID','item_id','shop_id'],axis=1)
test.head()

# Rearrange the raw data to be monthly average price by item-shop
# Scale Price
scaler = MinMaxScaler(feature_range=(0, 1))
sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()
df2.head()


# Merge data from average prices to specific item-shops in test data
price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)
price.head()

# Create x and y training sets from oldest data points
y_train = test['2015-10']
x_sales = test.drop(labels=['2015-10'],axis=1)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
x_prices = price.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
X = np.append(x_sales,x_prices,axis=2)

y = y_train.values.reshape((214200, 1))
print("Training Predictor Shape: ",X.shape)
print("Training Predictee Shape: ",y.shape)
del y_train, x_sales; gc.collect()

# Transform test set into numpy matrix
test = test.drop(labels=['2013-01'],axis=1)
x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

# Combine Price and Sales Df
test = np.append(x_test_sales,x_test_prices,axis=2)
del x_test_sales,x_test_prices, price; gc.collect()
print("Test Predictor Shape: ",test.shape)

# our defining our model
my_model = Sequential()
my_model.add(LSTM(units = 64, activation='tanh', input_shape = (X.shape[1], X.shape[2]), return_sequences=True))
my_model.add(Dropout(0.5))
my_model.add(LSTM(units= 32, activation='tanh'))
my_model.add(Dropout(0.5))
my_model.add(Dense(1))

my_model.compile(loss = 'mse',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
my_model.summary()

# from keras.utils import plot_model
# plot_model(my_model, to_file='mvar_lstm_model.png', show_shapes=True, show_layer_names=True)

save_best_only = ModelCheckpoint('mvar_lstm_model.dth', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)

save_best_only = ModelCheckpoint('lstm_model.dth', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

hist = my_model.fit(X_train,y_train,validation_data=(X_valid, y_valid), batch_size = 4096, epochs = 100, callbacks=[save_best_only])

print("Stopping")
# creating submission file
submission_pfs = my_model.predict(X)
# we will keep every value between 0 and 20
submission_pfs = submission_pfs.clip(0,20)
print("\Output Submission")
submission_pfs = pd.DataFrame(submission_pfs,columns=['item_cnt_month'])
submission_pfs.to_csv('submission_pfs.csv',index_label='ID')
print(submission_pfs.head())

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('mvar_lstm_loss.png')

# saving the history
import json
with open('file.json', 'w') as f:
    json.dump(hist.history, f)