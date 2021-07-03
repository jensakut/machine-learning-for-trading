#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import datetime
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(42)


def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def get_train_valid_data(X, y, train_idx, test_idx):
    x_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    x_val, y_val = X.iloc[test_idx, :], y.iloc[test_idx]
    return x_train, y_train, x_val, y_val


def calculate_splits(data, lookahead=21, train_test_ratio=5, n_splits=10, unstack=False):
    """
    todo docstring
    """
    if unstack:
        print('unstack {}'.format(unstack))
        data = data.unstack(unstack)

    print("Duplicate index: ".format(data[data.index.duplicated()]))
    nunique_rows = len(data.index)
    # there are train_test_ratio (5) splits needed for training and 1 for validation
    # and for shifting it n_splits times for one effective length, some space is needed as well
    effective_n_splits = (n_splits - 1) + (train_test_ratio + 1)
    effective_split_length = nunique_rows / effective_n_splits - lookahead
    # a training duration is effective split length times a test duration
    train_length = int(effective_split_length * train_test_ratio)
    test_length = int(effective_split_length)
    return train_length, test_length


def generate_random_prices(n=random.randint(365*1, 365*4)):
    stock_a = 1
    stock_b = 1
    stock_c = 1
    stock_d = 1
    prices = []
    date = datetime.date(year=2020, month=1, day=1)
    for i in range(n):
        prices.append([date, stock_a, stock_b, stock_c, stock_d])
        stock_a *= 1 + random.randrange(-90, 110) / 1000 / 24
        stock_b *= 1 + random.randrange(-100, 130) / 1000 / 24
        stock_c *= 1 + random.randrange(-101, 130) / 1000 / 24
        stock_d *= 1 + random.randrange(-106, 130) / 1000 / 24
        date += datetime.timedelta(days=1)
    prices = pd.DataFrame(data=prices, columns=['date', 'stock_A', 'stock_B', 'stock_C', 'stock_D'])
    prices.set_index('date', inplace=True)
    return prices


def reduce_footprint(df, floats=True, float_type='float32',
                     ints=True, int_type='int16',
                     objects=True,
                     except_cols=[]):
    if floats:
        change_datatype(df, except_cols=except_cols,
                        types=['float64'],
                        new_type=float_type)
    if ints:
        change_datatype(df, except_cols=except_cols,
                        types=['int64', 'int32'],
                        new_type=int_type)

    if objects:
        change_datatype(df, except_cols=except_cols,
                        types=['object'],
                        new_type='category')
    return df


def change_datatype(df, types, new_type, except_cols):
    cols = df.select_dtypes(include=types).columns.tolist()
    cols = list(set(cols) - set(except_cols))
    df[cols] = df[cols].astype(new_type)
    return df


if __name__ == "__main__":
    prices = generate_random_prices()
    n_splits = random.randint(3, 5)
    lookahead = random.randint(1, 21)
    train_test_ratio = random.randint(3, 8)

    train_length, test_length = calculate_splits(prices, lookahead=lookahead,
                                                 train_test_ratio=5, n_splits=n_splits,
                                                 unstack=False)

    # time-series cross-validation
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              lookahead=lookahead,
                              test_period_length=test_length,
                              train_period_length=train_length)

    fig, ax = plt.subplots(nrows=n_splits, sharex='col')
    print('prices dates {}:{}'.format(prices.index.get_level_values('date').min(),
                                      prices.index.get_level_values('date').max()))

    for i, (train_idx, test_idx) in enumerate(cv.split(X=prices)):
        print(i)
        x_train, y_train, x_val, y_val = get_train_valid_data(prices, prices, train_idx, test_idx)
        print('x_train dates {}:{}'.format(x_train.index.get_level_values('date').min(),
                                         x_train.index.get_level_values('date').max()))
        print('y_train dates {}:{}'.format(y_train.index.get_level_values('date').min(),
                                         y_train.index.get_level_values('date').max()))
        print('x_val dates {}:{}'.format(x_val.index.get_level_values('date').min(),
                                         x_val.index.get_level_values('date').max()))
        print('y_val dates {}:{}'.format(y_val.index.get_level_values('date').min(),
                                         y_val.index.get_level_values('date').max()))
        ax[i].plot(x_train.index.get_level_values('date'),
                 x_train)
        ax[i].plot(x_val.index.get_level_values('date'),
                    x_val)
    plt.show()


