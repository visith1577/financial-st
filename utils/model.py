import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import torch


class Model(nn.Module):
    def __init__(self,
                 input_size=4,
                 hidden_layer_size=32,
                 num_layers=2,
                 output_size=4,
                 dropout=0.2,
                 conv1d_out_channels=32,
                 conv1d_kernel_size=3,
                 conv2d_out_channels=64,
                 conv2d_kernel_size=(3, 1)):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm_input_size = 18 * conv2d_out_channels

        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv1d_out_channels,
                                kernel_size=conv1d_kernel_size)
        self.conv1d_1 = nn.Conv1d(in_channels=conv1d_out_channels, out_channels=conv1d_out_channels,
                                  kernel_size=conv1d_kernel_size)
        self.conv1d_activation = nn.ReLU()
        self.conv1d_batchnorm = nn.BatchNorm1d(conv1d_out_channels)
        self.conv2d = nn.Conv2d(in_channels=conv1d_out_channels, out_channels=conv2d_out_channels,
                                kernel_size=conv2d_kernel_size, padding=(1, 0))
        self.con2d_1 = nn.Conv2d(in_channels=conv2d_out_channels, out_channels=conv2d_out_channels,
                                 kernel_size=conv2d_kernel_size, padding=(1, 0))
        self.con2d_2 = nn.Conv2d(in_channels=conv2d_out_channels, out_channels=conv2d_out_channels,
                                 kernel_size=conv2d_kernel_size, padding=(1, 0))
        self.conv2d_activation = nn.ReLU()

        self.linear_1 = nn.Linear(self.lstm_input_size, hidden_layer_size)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_layer_size * num_layers, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.conv1d_activation(x)

        x = x.unsqueeze(-1)

        x = self.conv2d(x)
        x = self.conv2d_activation(x)
        x = self.con2d_1(x)
        x = self.conv2d_activation(x)
        x = self.con2d_2(x)
        x = self.conv2d_activation(x)

        x = x.view(batchsize, -1)

        x = self.linear_1(x)
        x = self.relu(x)

        x = x.unsqueeze(-1).permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.dropout(x)
        predictions = self.dense(x)

        return predictions


async def run_epoch(dataloader, model, optimizer, criterion, scheduler, is_training=False, device=torch.device('mps')):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):

        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


async def train_model(model, dataset_train, dataset_val, batch_size=32, device=torch.device('mps'), epochs=100, lr=1e-3,
                      scheduler=10):
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler, gamma=0.1)

    my_bar = st.progress(0, text="Starting")

    for epoch in range(epochs):
        loss_train, lr_train = await run_epoch(train_dataloader, model, optimizer, criterion, scheduler,
                                               is_training=True)
        loss_val, lr_val = await run_epoch(val_dataloader, model, optimizer, criterion, scheduler)
        scheduler.step()

        my_bar.progress((epoch + 1) / epochs,
                        text='Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'.format(epoch + 1, epochs,
                                                                                                loss_train, loss_val,
                                                                                                lr_train))

    return model


async def launch_model(model, dataset_train, dataset_val, chunk_of_unseen_data, pred_days=1, batch_size=32,
                       device=torch.device('mps')):
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model.eval()

    predicted_train = np.empty((1, dataset_val.y.shape[1]), dtype=float)

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    predicted_train = np.delete(predicted_train, 0, axis=0)

    predicted_val = np.empty((1, dataset_val.y.shape[1]), dtype=float)

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    predicted_val = np.delete(predicted_val, 0, axis=0)

    Day_to_predict = pred_days
    data_x_unseen = np.array([chunk_of_unseen_data.T])

    predicted_day = np.empty((1, dataset_val.y.shape[1], 1), dtype=float)

    for _ in range(Day_to_predict):
        model.eval()

        x = torch.tensor(data_x_unseen).float().to(device)
        prediction = model(x)
        prediction = prediction.cpu().detach().numpy()
        predictions = prediction[0].reshape(1, 1, -1)

        data_x_unseen = np.concatenate((data_x_unseen, predictions), axis=1)
        data_x_unseen = np.delete(data_x_unseen, 0, axis=1)

        prediction = np.expand_dims(prediction, axis=2)
        predicted_day = np.concatenate((predicted_day, prediction), axis=2)

    predicted_day = np.delete(predicted_day, 0, axis=2)

    return predicted_train, predicted_val, predicted_day


async def graph_preds(data_date, num_data_points, predicted_train, predicted_val, predicted_day, data_close_price,
                      scaler, split_index, pred_days=5, window_size=20):
    predicted_train_open = predicted_train.T[0]
    predicted_val_open = predicted_val.T[0]
    predicted_train_high = predicted_train.T[1]
    predicted_val_high = predicted_val.T[1]
    predicted_train_low = predicted_train.T[2]
    predicted_val_low = predicted_val.T[2]
    predicted_train_close = predicted_train.T[3]
    predicted_val_close = predicted_val.T[3]

    to_plot_data_y_train_pred_open = np.zeros(num_data_points)
    to_plot_data_y_train_pred_high = np.zeros(num_data_points)
    to_plot_data_y_train_pred_low = np.zeros(num_data_points)
    to_plot_data_y_train_pred_close = np.zeros(num_data_points)

    to_plot_data_y_val_pred_open = np.zeros(num_data_points)
    to_plot_data_y_val_pred_high = np.zeros(num_data_points)
    to_plot_data_y_val_pred_low = np.zeros(num_data_points)
    to_plot_data_y_val_pred_close = np.zeros(num_data_points)

    predicted_train_open = np.array([predicted_train_open]).T
    predicted_train_high = np.array([predicted_train_high]).T
    predicted_train_low = np.array([predicted_train_low]).T
    predicted_train_close = np.array([predicted_train_close]).T

    predicted_val_open = np.array([predicted_val_open]).T
    predicted_val_high = np.array([predicted_val_high]).T
    predicted_val_low = np.array([predicted_val_low]).T
    predicted_val_close = np.array([predicted_val_close]).T

    predicted_train_open = scaler.inverse_transform(predicted_train_open).T[0]
    predicted_train_high = scaler.inverse_transform(predicted_train_high).T[0]
    predicted_train_low = scaler.inverse_transform(predicted_train_low).T[0]
    predicted_train_close = scaler.inverse_transform(predicted_train_close).T[0]

    predicted_val_open = scaler.inverse_transform(predicted_val_open).T[0]
    predicted_val_high = scaler.inverse_transform(predicted_val_high).T[0]
    predicted_val_low = scaler.inverse_transform(predicted_val_low).T[0]
    predicted_val_close = scaler.inverse_transform(predicted_val_close).T[0]

    to_plot_data_y_train_pred_open[window_size:split_index + window_size] = predicted_train_open
    to_plot_data_y_train_pred_high[window_size:split_index + window_size] = predicted_train_high
    to_plot_data_y_train_pred_low[window_size:split_index + window_size] = predicted_train_low
    to_plot_data_y_train_pred_close[window_size:split_index + window_size] = predicted_train_close

    to_plot_data_y_val_pred_open[split_index + window_size:] = predicted_val_open
    to_plot_data_y_val_pred_high[split_index + window_size:] = predicted_val_high
    to_plot_data_y_val_pred_low[split_index + window_size:] = predicted_val_low
    to_plot_data_y_val_pred_close[split_index + window_size:] = predicted_val_close

    to_plot_data_y_train_pred_close = np.where(to_plot_data_y_train_pred_close == 0, False,
                                               to_plot_data_y_train_pred_close)

    to_plot_data_y_val_pred_close = np.where(to_plot_data_y_val_pred_close == 0, False, to_plot_data_y_val_pred_close)

    data_close_price = data_close_price.reshape(-1, 1)

    fag = go.Figure()
    fag.add_trace(go.Scatter(name="Actual prices", x=data_date, y=data_close_price.T[0], line=dict(color="#fd7f20")))
    fag.add_trace(go.Scatter(name="Predicted prices (train)", x=data_date, y=to_plot_data_y_train_pred_close,
                             line=dict(color="#fdb750")))
    fag.add_trace(go.Scatter(name="Predicted prices (validation)", x=data_date, y=to_plot_data_y_val_pred_close,
                             line=dict(color="#d3d3cb")))
    fag.update_layout(title_text="Data prediction")
    fag.update_xaxes(showgrid=True, ticklabelmode="period")
    fag.update_layout(xaxis_rangeslider_visible=False)

    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range - 1)

    to_plot_data_y_val_pred_open = np.zeros(plot_range - 1)
    to_plot_data_y_val_pred_high = np.zeros(plot_range - 1)
    to_plot_data_y_val_pred_low = np.zeros(plot_range - 1)
    to_plot_data_y_val_pred_close = np.zeros(plot_range - 1)

    to_plot_data_y_test_pred_open = np.zeros(plot_range - 1)
    to_plot_data_y_test_pred_high = np.zeros(plot_range - 1)
    to_plot_data_y_test_pred_low = np.zeros(plot_range - 1)
    to_plot_data_y_test_pred_close = np.zeros(plot_range - 1)

    data_y_val = data_close_price.T[0]
    to_plot_data_y_val[:plot_range - 1] = data_y_val[-plot_range + 1:]

    to_plot_data_y_val_pred_open[:plot_range - 1] = predicted_val_open[-plot_range + 1:]
    to_plot_data_y_val_pred_high[:plot_range - 1] = predicted_val_high[-plot_range + 1:]
    to_plot_data_y_val_pred_low[:plot_range - 1] = predicted_val_low[-plot_range + 1:]
    to_plot_data_y_val_pred_close[:plot_range - 1] = predicted_val_close[-plot_range + 1:]

    plot_date_test = data_date[-plot_range + 1:]

    PREDI_VAL_open = []
    PREDI_VAL_high = []
    PREDI_VAL_low = []
    PREDI_VAL_close = []

    Day_to_predict = pred_days
    prediction = predicted_day.T

    for day_pred in range(Day_to_predict):
        prediction_open = prediction[day_pred][0]
        prediction_high = prediction[day_pred][1]
        prediction_low = prediction[day_pred][2]
        prediction_close = prediction[day_pred][3]

        pred_conv_open = np.array([prediction_open]).T
        pred_conv_high = np.array([prediction_high]).T
        pred_conv_low = np.array([prediction_low]).T
        pred_conv_close = np.array([prediction_close]).T

        pred_conv_open = scaler.inverse_transform(pred_conv_open).T[0]
        pred_conv_high = scaler.inverse_transform(pred_conv_high).T[0]
        pred_conv_low = scaler.inverse_transform(pred_conv_low).T[0]
        pred_conv_close = scaler.inverse_transform(pred_conv_close).T[0]

        to_plot_data_y_val = np.append(to_plot_data_y_val, 0)

        to_plot_data_y_val_pred_open = np.append(to_plot_data_y_val_pred_open, 0)
        to_plot_data_y_val_pred_high = np.append(to_plot_data_y_val_pred_high, 0)
        to_plot_data_y_val_pred_low = np.append(to_plot_data_y_val_pred_low, 0)
        to_plot_data_y_val_pred_close = np.append(to_plot_data_y_val_pred_close, 0)

        to_plot_data_y_test_pred_open = np.append(to_plot_data_y_test_pred_open, pred_conv_open)
        to_plot_data_y_test_pred_high = np.append(to_plot_data_y_test_pred_high, pred_conv_high)
        to_plot_data_y_test_pred_low = np.append(to_plot_data_y_test_pred_low, pred_conv_low)
        to_plot_data_y_test_pred_close = np.append(to_plot_data_y_test_pred_close, pred_conv_close)

        PREDI_VAL_open.append(pred_conv_open)
        PREDI_VAL_high.append(pred_conv_high)
        PREDI_VAL_low.append(pred_conv_low)
        PREDI_VAL_close.append(pred_conv_close)

        new_day = plot_date_test[-1] + np.timedelta64(1, 'D')
        plot_date_test = np.append(plot_date_test, new_day)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, False, to_plot_data_y_val)

        to_plot_data_y_val_pred_open = np.where(to_plot_data_y_val_pred_open == 0, False, to_plot_data_y_val_pred_open)
        to_plot_data_y_val_pred_high = np.where(to_plot_data_y_val_pred_high == 0, False, to_plot_data_y_val_pred_high)
        to_plot_data_y_val_pred_low = np.where(to_plot_data_y_val_pred_low == 0, False, to_plot_data_y_val_pred_low)
        to_plot_data_y_val_pred_close = np.where(to_plot_data_y_val_pred_close == 0, False,
                                                 to_plot_data_y_val_pred_close)

        to_plot_data_y_test_pred_open = np.where(to_plot_data_y_test_pred_open == 0, False,
                                                 to_plot_data_y_test_pred_open)
        to_plot_data_y_test_pred_high = np.where(to_plot_data_y_test_pred_high == 0, False,
                                                 to_plot_data_y_test_pred_high)
        to_plot_data_y_test_pred_low = np.where(to_plot_data_y_test_pred_low == 0, False, to_plot_data_y_test_pred_low)
        to_plot_data_y_test_pred_close = np.where(to_plot_data_y_test_pred_close == 0, False,
                                                  to_plot_data_y_test_pred_close)

    PREDI_VAL_open = np.array(PREDI_VAL_open)
    PREDI_VAL_high = np.array(PREDI_VAL_high)
    PREDI_VAL_low = np.array(PREDI_VAL_low)
    PREDI_VAL_close = np.array(PREDI_VAL_close)

    fog = go.Figure()
    fog.add_trace(go.Scatter(name="Actual prices", x=plot_date_test, y=to_plot_data_y_val, line=dict(color="#fd7f20")))
    fog.add_trace(go.Scatter(name="Past predicted prices", x=plot_date_test, y=to_plot_data_y_val_pred_close,
                             line=dict(color="#fdb750")))
    fog.add_trace(go.Scatter(name="Predicted price for next day", x=plot_date_test, y=to_plot_data_y_test_pred_close,
                             marker={
                                 "symbol": "circle",
                             }, line=dict(color="#d3d3cb")))
    fog.add_trace(go.Candlestick(name="Train Data Close", x=plot_date_test,
                                 open=to_plot_data_y_val_pred_open,
                                 high=to_plot_data_y_val_pred_high,
                                 low=to_plot_data_y_val_pred_low,
                                 close=to_plot_data_y_val_pred_close,
                                 increasing_line={
                                     "color": 'cyan'
                                 }))
    fog.add_trace(go.Candlestick(name="Predict Data Close", x=plot_date_test,
                                 open=to_plot_data_y_test_pred_open,
                                 high=to_plot_data_y_test_pred_high,
                                 low=to_plot_data_y_test_pred_low,
                                 close=to_plot_data_y_test_pred_close))
    fog.update_layout(title_text="Stock Forecasting")
    fog.update_xaxes(showgrid=True, ticklabelmode="period")
    fog.update_layout(xaxis_rangeslider_visible=False)

    PREDI_VAL_open = np.append(to_plot_data_y_val, PREDI_VAL_open.T[0])
    PREDI_VAL_high = np.append(to_plot_data_y_val, PREDI_VAL_high.T[0])
    PREDI_VAL_low = np.append(to_plot_data_y_val, PREDI_VAL_low.T[0])
    PREDI_VAL_close = np.append(to_plot_data_y_val, PREDI_VAL_close.T[0])

    PREDI_VALS_open = [i for i in PREDI_VAL_open if i != 0]
    PREDI_VALS_high = [i for i in PREDI_VAL_high if i != 0]
    PREDI_VALS_low = [i for i in PREDI_VAL_low if i != 0]
    PREDI_VALS_close = [i for i in PREDI_VAL_close if i != 0]

    predi_time = plot_date_test[-(window_size + Day_to_predict):]

    FinalPred = pd.DataFrame(
        {'Date': predi_time, 'Open': PREDI_VALS_open, 'High': PREDI_VALS_high, 'Low': PREDI_VALS_low,
         'Close': PREDI_VALS_close})
    FinalPred = FinalPred.set_index('Date')
    FinalPred = FinalPred.tail(Day_to_predict)

    return fag, fog, FinalPred
