import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

import gc

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        # out, _ = self.lstm_1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm_1(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = nn.Dropout(p=0.2)(out)
        out, _ = self.lstm_2(out)
        out = nn.Dropout(p=0.2)(out)
        out, _ = self.lstm_3(out)
        out = nn.Dropout(p=0.2)(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
        # return out[:, -1, :]

class Model(nn.Module):
    def __init__(self, city_label_size=31, item_cat_size=84, shop_id_size=60):
        super().__init__()

        def embedding_network(input_size, emb_size=64):
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, emb_size)
            )

        self.city_label_size = city_label_size
        self.item_cat_size = item_cat_size
        self.shop_id_size = shop_id_size

        self.city_embedding = embedding_network(self.city_label_size, emb_size=64)
        self.item_cat_embedding = embedding_network(self.item_cat_size, emb_size=64)
        self.shop_id_embedding = embedding_network(self.shop_id_size, emb_size=64)
        self.lstm = RNN(input_size=1, hidden_size=64, num_layers=2)

        self.fcnn = nn.Sequential(
                        nn.Linear(1+(64*3), 32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(32, 1)
        )

    def forward(self, item_count_ts, item_cat_vector, shop_id_vector, city_label_vector):
        city_emb = self.city_embedding(city_label_vector)
        item_cat_emb = self.item_cat_embedding(item_cat_vector)
        shop_id_emb = self.shop_id_embedding(shop_id_vector)
        lstm_out = self.lstm(item_count_ts.unsqueeze(2))

        # # concat all
        concat_vector = torch.cat((lstm_out, city_emb), 1)
        return self.fcnn(concat_vector)


class SalesDataset(Dataset):
    def __init__(self, sales_train, sales_clean, test, is_train):
        self.sales_train = pd.read_csv(sales_train)
        self.sales_clean = pd.read_csv(sales_clean)
        self.test = pd.read_csv(test)
        self.is_train = is_train

        # removing shop id and item id which are not in test
        test_shop_ids = self.test['shop_id'].unique()
        test_item_ids = self.test['item_id'].unique()
        # Only shops that exist in test set.
        self.sales_train = self.sales_train[self.sales_train['shop_id'].isin(test_shop_ids)]
        # Only items that exist in test set.
        self.sales_train = self.sales_train[self.sales_train['item_id'].isin(test_item_ids)]
        self.sales_train['date'] = pd.to_datetime(self.sales_train['date'], format='%d.%m.%Y')
        # now we will create a pivot tabel by going so we get our data in desired form
        # we want get total count value of an item over the whole month for a shop
        # That why we made shop_id and item_id our indices and date_block_num our column
        # the value we want is item_cnt_day and used sum as aggregating function
        self.dataset = self.sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'],
                                                    columns=['date_block_num'], fill_value=0, aggfunc='sum')
        self.dataset.reset_index(inplace=True)

        if not self.is_train:
            # Now we will merge our pivot table with the test_data because we want to keep the data of items we have predict
            self.dataset = pd.merge(self.test, self.dataset, on=['item_id', 'shop_id'], how='left')

        extra = self.sales_clean[['shop_id', 'item_id', 'city_label', 'item_category_id', 'holidays_in_month']]
        extra = extra.drop_duplicates(['item_id', 'shop_id'], keep='last')
        # price = self.sales_train[['shop_id', 'item_id', 'item_price']]
        # price = price.drop_duplicates(['item_id', 'shop_id'], keep='last')

        # merging extra
        self.dataset = pd.merge(self.dataset, extra, on=['item_id', 'shop_id'], how='left')
        # self.dataset = pd.merge(self.dataset, price, on=['item_id', 'shop_id'], how='left')
        self.dataset.fillna(0, inplace=True)
        self.dataset.clip(0, )

        self.max_item_cat = int(max(self.dataset['item_category_id'])) + 1
        self.max_shop_id = int(max(self.dataset['shop_id'])) + 1
        self.max_city_label = int(max(self.dataset['city_label'])) + 1

        del self.sales_train
        del self.sales_clean
        gc.collect()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

        # one hot encoding for item_cat
        item_cat_vector = [0 for _ in range(self.max_item_cat)]
        item_cat_vector[int(row['item_category_id'])] = 1

        # one hot encoding for shop_id
        shop_id_vector = [0 for _ in range(self.max_shop_id)]
        shop_id_vector[int(row['shop_id'])] = 1

        # one hot encoding for city_label
        city_label_vector = [0 for _ in range(self.max_city_label)]
        city_label_vector[int(row['city_label'])] = 1

        if self.is_train:
            item_count_ts = list(row[3: 36])
            target = row[37]
        else:
            item_count_ts = list(row[4: 37])
            target = 0


        item_count_ts = torch.FloatTensor(item_count_ts)
        item_cat_vector = torch.FloatTensor(item_cat_vector)
        shop_id_vector = torch.FloatTensor(shop_id_vector)
        city_label_vector = torch.FloatTensor(city_label_vector)
        target = torch.FloatTensor([target])

        return item_count_ts, item_cat_vector, shop_id_vector, city_label_vector, target

    def get_test_data(self):
        return self.test

is_cuda = torch.cuda.is_available()
print("Is cuda availiable: ", is_cuda)

train_sales_dataset = SalesDataset(sales_train='../data/sales_train.csv',
                             sales_clean='../data/cleaned_sales.csv',
                             test='../data/test.csv',
                             is_train=True)

test_sales_dataset = SalesDataset(sales_train='../data/sales_train.csv',
                             sales_clean='../data/cleaned_sales.csv',
                             test='../data/test.csv',
                             is_train=False)

train_dataloader = DataLoader(train_sales_dataset, batch_size=4096, num_workers=4)
test_dataloader = DataLoader(test_sales_dataset, batch_size=4096, num_workers=4)

model = Model()
print(model)
no_epochs = 10
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

if is_cuda:
    model.cuda()

model.train()
for epoch in range(no_epochs):

    losses = []
    for idx, (item_count_ts, item_cat_vector, shop_id_vector, city_label_vector, target) in enumerate(train_dataloader):

        optimizer.zero_grad()
        if is_cuda:
            item_count_ts, item_cat_vector, shop_id_vector, city_label_vector, target = \
                item_count_ts.cuda(), item_cat_vector.cuda(), shop_id_vector.cuda(), city_label_vector.cuda(), target.cuda()

        pred = model(item_count_ts, item_cat_vector, shop_id_vector, city_label_vector)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print("Epoch: {} MSE Loss: {}".format(epoch+1, sum(losses)/len(losses)))

torch.save(model.state_dict(), 'hybrid_lstm')

model.eval()
submission_pfs = np.array([])
for idx, (item_count_ts, item_cat_vector, shop_id_vector, city_label_vector, target) in enumerate(test_dataloader):

    if is_cuda:
        item_count_ts, item_cat_vector, shop_id_vector, city_label_vector, target = \
            item_count_ts.cuda(), item_cat_vector.cuda(), shop_id_vector.cuda(), city_label_vector.cuda(), target.cuda()

    pred = model(item_count_ts, item_cat_vector, shop_id_vector, city_label_vector)
    submission_pfs = np.append(submission_pfs, pred.data.cpu().numpy())

# creating submission file
# we will keep every value between 0 and 20
submission_pfs = submission_pfs.clip(0,20)
test = train_sales_dataset.get_test_data()
# creating dataframe with required columns
submission = pd.DataFrame({'ID': test['ID'],'item_cnt_month':submission_pfs.ravel()})
# creating csv file from dataframe
submission.to_csv('hyb_sub_pfs.csv',index = False)