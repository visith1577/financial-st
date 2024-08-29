import time

import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


class QLearningTrader:
    def __init__(self, num_actions, num_features, learning_rate, discount_factor, exploration_prob):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        self.q_table = np.zeros((num_actions, num_features))
        self.current_state = None
        self.current_action = None

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            action = np.random.choice(self.num_actions)
        else:
            feature_index = np.argmax(state)
            action = np.argmax(self.q_table[:, feature_index])
        return action

    def observe_realtime_data(self, prices_to_predict):
        self.current_state = np.array(
            [prices_to_predict[0], prices_to_predict[1], prices_to_predict[2], prices_to_predict[3]])

    def observe_next_state(self, next_prices_to_predict):
        self.current_state = np.array([next_prices_to_predict[0], next_prices_to_predict[1], next_prices_to_predict[2],
                                       next_prices_to_predict[3]])

    def take_action(self, action, reward):
        if self.current_action is not None:
            feature_index = np.argmax(self.current_state)
            current_q_value = self.q_table[self.current_action, feature_index]
            new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                    reward + self.discount_factor * np.max(self.q_table[:, feature_index]))
            self.q_table[self.current_action, feature_index] = new_q_value
        self.current_state = None
        self.current_action = action


def calculate_reward(action, current_price, next_price):
    if action == 0:
        return 1.0 if next_price > current_price else -1.0
    elif action == 1:
        return 1.0 if next_price < current_price else -1.0
    else:
        return 1.0 if next_price > current_price else -1.0 if next_price < current_price else 0.0


def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d', interval='1m')
    return data


def calculate_profit_loss(initial_balance, suggested_action, current_price, next_price, quantity):
    if suggested_action == 'Buy':
        return (next_price - current_price) * quantity
    elif suggested_action == 'Sell':
        return (current_price - next_price) * quantity
    else:
        return 0.0


def prepare_predict_data(store_data):
    prices_to_predict_array = [store_data[0], max(store_data), min(store_data), store_data[-1]]
    prices_to_predict = np.array(prices_to_predict_array).reshape(-1, 1)
    return prices_to_predict.T[0]


def get_latest_price(data):
    latest_price = []
    if not data.empty:
        latest_price.append(data['Close'].iloc[-1])
        latest_prices = np.array(latest_price).reshape(-1, 1)
        return latest_prices
    else:
        return None


def update_data(
        symbol,
        price_placeholder,
        initial_balance,
        quantity,
        num_iterations,
        learning_rate,
        discount_factor,
        exploration_prob,
        time_place,
        update_interval=15):
    start_time = time.time()
    store_data = []
    action_hold = []
    profit_hold = []
    f = 0
    num_actions = 3
    num_features = 5

    q_trader = QLearningTrader(num_actions, num_features, learning_rate, discount_factor, exploration_prob)

    for i in range(num_iterations):
        data = get_stock_price(symbol)
        latest_price = get_latest_price(data)
        price = latest_price[0][0]

        if price is not None:
            price_pr = str("$" + str(round(price, 2)) + " USD")
            price_placeholder.header(f" :green[_{price_pr}_]")
            store_data.append(price)
            elapsed_time = time.time() - start_time
            time_place.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

            if elapsed_time > update_interval:
                start_time = time.time()
                prices_to_predict = prepare_predict_data(store_data)

                q_trader.observe_realtime_data(prices_to_predict)
                action = q_trader.choose_action(q_trader.current_state)
                current_close = q_trader.current_state[3]
                store_data_next = []
                for _ in range(5):
                    # data = get_stock_price(symbol)
                    # latest_price = get_latest_price(data)
                    # price = latest_price[0][0]
                    if price is not None:
                        store_data_next.append(price)
                    time.sleep(1)
                prices_to_predict_next = prepare_predict_data(store_data_next)
                q_trader.observe_next_state(prices_to_predict_next)
                reward = calculate_reward(action, current_close, q_trader.current_state[3])
                q_trader.take_action(action, reward)

                final_data = get_stock_price(symbol)
                final_price = get_latest_price(final_data)
                final_real_time_data = final_price[0][0]

                suggested_action = ["Buy", "Sell", "Hold"][
                    np.argmax(q_trader.q_table[:, np.argmax(q_trader.current_state)])]
                f += 1
                profit = calculate_profit_loss(initial_balance, suggested_action, current_close, final_real_time_data,
                                               quantity)
                suggested_action = suggested_action + f"_{f}"

                action_hold.append(suggested_action)
                profit_hold.append(profit)

                # st.write(prices_to_predict)
                store_data.clear()

        time.sleep(0.1)

    final_data = get_stock_price(symbol)
    final_price = get_latest_price(final_data)
    final_real_time_data = final_price[0][0]

    df = pd.DataFrame(list(zip(action_hold, profit_hold)), columns=['action', 'reward'])

    final_suggested_action = ["Buy", "Sell", "Hold"][np.argmax(q_trader.q_table[:, np.argmax(q_trader.current_state)])]
    final_profit = calculate_profit_loss(initial_balance, final_suggested_action, current_close, final_real_time_data,
                                         quantity)
    st.write(f"Final Suggested Action: {final_suggested_action}, Final Profit: {final_profit}")

    action_np = np.array(df['action'])
    reward_np = np.array(df['reward'])

    colors = np.where(reward_np < 0, '#ff0000', '#00FF00')

    fig = go.Figure(data=[go.Bar(x=action_np, y=reward_np, marker_color=colors)])
    return fig
