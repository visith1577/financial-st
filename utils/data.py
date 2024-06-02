import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class FinanceDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, axis=2)
        self.x = x.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


async def fetch_data(df, symbol):
    data_date = df.index.to_numpy().reshape(1, -1)
    data_open_price = df['Open'].to_numpy().reshape(1, -1)
    data_high_price = df['High'].to_numpy().reshape(1, -1)
    data_low_price = df['Low'].to_numpy().reshape(1, -1)
    data_close_price = df['Close'].to_numpy().reshape(1, -1)
    df_data = np.concatenate((data_date, data_open_price, data_high_price, data_low_price, data_close_price), axis=0)
    return df_data


async def normalized_data(asset_data):
    data_open_price = asset_data[1].reshape(-1, 1)
    data_high_price = asset_data[2].reshape(-1, 1)
    data_low_price = asset_data[3].reshape(-1, 1)
    data_close_price = asset_data[4].reshape(-1, 1)

    datas = np.concatenate((data_open_price, data_high_price, data_low_price, data_close_price), axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(np.array(datas))
    norm_data = norm_data.T[0]

    int_size = data_open_price.shape[0]

    data_open_price_norm = norm_data[:int_size]
    data_high_price_norm = norm_data[int_size:(int_size * 2)]
    data_low_price_norm = norm_data[(int_size * 2):(int_size * 3)]
    data_close_price_norm = norm_data[(int_size * 3):(int_size * 4)]

    data_open_price_norm = data_open_price_norm.reshape(1, -1)
    data_high_price_norm = data_high_price_norm.reshape(1, -1)
    data_low_price_norm = data_low_price_norm.reshape(1, -1)
    data_close_price_norm = data_close_price_norm.reshape(1, -1)

    norm_data = np.concatenate((data_open_price_norm, data_high_price_norm, data_low_price_norm, data_close_price_norm),
                               axis=0)

    return norm_data, scaler


async def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


async def prepare_data_y(x, window_size):
    output = x[window_size:]
    return output


async def data_on_percent(datas, percent):
    data = datas[0]
    data_x, data_x_unseen = await prepare_data_x(data, window_size=20)
    data_y = await prepare_data_y(data, window_size=20)

    split_index = int(data_y.shape[0] * percent)
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    dataset_train = FinanceDataset(data_x_train, data_y_train)
    dataset_val = FinanceDataset(data_x_val, data_y_val)

    dataset_train.y = dataset_train.y.reshape(-1, 1)
    dataset_val.y = dataset_val.y.reshape(-1, 1)

    data_x_unseen = data_x_unseen.reshape(1, -1)
    datas = np.delete(datas, 0, axis=0)

    for data in datas:
        data_x_, data_x_unseen_ = await prepare_data_x(data, window_size=20)
        data_y_ = await prepare_data_y(data, window_size=20)

        data_x_train_ = data_x_[:split_index]
        data_x_val_ = data_x_[split_index:]
        data_y_train_ = data_y_[:split_index]
        data_y_val_ = data_y_[split_index:]

        dataset_train_ = FinanceDataset(data_x_train_, data_y_train_)
        dataset_val_ = FinanceDataset(data_x_val_, data_y_val_)

        dataset_train_.y = dataset_train_.y.reshape(-1, 1)
        dataset_val_.y = dataset_val_.y.reshape(-1, 1)

        data_x_unseen_ = data_x_unseen_.reshape(1, -1)

        dataset_train.x = np.concatenate((dataset_train.x, dataset_train_.x), axis=2)
        dataset_val.x = np.concatenate((dataset_val.x, dataset_val_.x), axis=2)
        dataset_train.y = np.concatenate((dataset_train.y, dataset_train_.y), axis=1)
        dataset_val.y = np.concatenate((dataset_val.y, dataset_val_.y), axis=1)
        data_x_unseen = np.concatenate((data_x_unseen, data_x_unseen_), axis=0)

    return data_x_unseen, dataset_train, dataset_val, split_index
