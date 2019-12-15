import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from math import sqrt
from numpy import loadtxt
from itertools import product
#from tqdm import tqdm
from sklearn import preprocessing
from xgboost import plot_tree
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

sales_train = pd.read_csv('../data/sales_train.csv')
items = pd.read_csv('../data/items.csv')
shops = pd.read_csv('../data/shops.csv')
item_categories = pd.read_csv('../data/item_categories.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales_train['date_block_num'].unique():
    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



# Aggregations
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)

trainset = pd.merge(grid,trainset,how='left',on=index_cols)
trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)


# Get category id
trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
trainset.to_csv('trainset_with_grid.csv')

print(trainset.head())


# Extract features and target we want
baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']
train = trainset[baseline_features]
# Remove pandas index column
train = train.set_index('shop_id')
train.item_cnt_month = train.item_cnt_month.astype(int)
train['item_cnt_month'] = train.item_cnt_month.fillna(0).clip(0,20)
# Save train set to file
train.to_csv('train.csv')


dataset = loadtxt('train.csv', delimiter="," ,skiprows=1, dtype = int)
trainx = dataset[:, 0:4]
trainy = dataset[:, 4]

test_dataset = loadtxt('data/test.csv', delimiter="," ,skiprows=1, usecols = (1,2), dtype=int)
test_df = pd.DataFrame(test_dataset, columns = ['shop_id', 'item_id'])

# Make test_dataset pandas data frame, add category id and date block num, then convert back to numpy array and predict
merged_test = pd.merge(test_df, items, on = ['item_id'])[['shop_id','item_id','item_category_id']]
merged_test['date_block_num'] = 33
merged_test.set_index('shop_id')
print(merged_test.head(3))


model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
model.fit(trainx, trainy, eval_metric='rmse')
preds = model.predict(merged_test.values)

df = pd.DataFrame(preds, columns = ['item_cnt_month'])
df['ID'] = df.index
df = df.set_index('ID')
df.to_csv('simple_xgb.csv')






