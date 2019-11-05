import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import keras

shops = pd.read_csv("data/shops.csv")
items = pd.read_csv("data/items.csv")
item_categories = pd.read_csv("data/item_categories.csv")
test = pd.read_csv("data/test.csv")
sales_train = pd.read_csv("data/sales_train.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# removing shop id and item id which are not in test
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
# Only shops that exist in test set.
sales_train = sales_train[sales_train['shop_id'].isin(test_shop_ids)]
# Only items that exist in test set.
sales_train = sales_train[sales_train['item_id'].isin(test_item_ids)]

sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
#now we will create a pivot tabel by going so we get our data in desired form 
#we want get total count value of an item over the whole month for a shop 
# That why we made shop_id and item_id our indices and date_block_num our column 
# the value we want is item_cnt_day and used sum as aggregating function 
dataset = sales_train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')

# lets reset our indices, so that data should be in way we can easily manipulate
dataset.reset_index(inplace = True)

# Now we will merge our pivot table with the test_data because we want to keep the data of items we have
# predict
dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')

# lets fill all NaN values with 0
# dataset.fillna(0,inplace = True)
dataset.fillna(0, inplace=True)
dataset.clip(0, )

# we will drop shop_id and item_id because we do not need them
# we are teaching our model how to generate the next sequence 
dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

# X we will keep all columns execpt the last one 
X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)
# the last column is our label
y_train = dataset.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

# lets have a look on the shape 
print(X_train.shape,y_train.shape,X_test.shape)

# importing libraries required for our model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import Adam 

epochs = 100
# our defining our model 
my_model = Sequential()
my_model.add(LSTM(units = 64, activation='tanh', input_shape = (33,1), return_sequences=True))
my_model.add(Dropout(0.5))
my_model.add(LSTM(units= 32, activation='tanh'))
my_model.add(Dropout(0.5))
my_model.add(Dense(1))
# opt = Adam(lr=1e-3, decay=1e-3/epochs)
my_model.compile(loss = 'mse',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
my_model.summary()


hist = my_model.fit(X_train,y_train,validation_split=0.2, batch_size = 4096,epochs = epochs)

# creating submission file 
submission_pfs = my_model.predict(X_test)
# we will keep every value between 0 and 20
submission_pfs = submission_pfs.clip(0,20)
# creating dataframe with required columns 
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})
# creating csv file from dataframe
submission.to_csv('sub_pfs.csv',index = False)

plt.plot(hist.history['loss'])
plt.title('train loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.savefig('train_loss.png')

# saving the history
import json
with open('file.json', 'w') as f:
    json.dump(hist.history, f)