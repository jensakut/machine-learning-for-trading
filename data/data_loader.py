import warnings
from datetime import time
from glob import glob

from utils import reduce_footprint

warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')


def clean_df(df):
    df = df.droplevel(['symbol', 'base'])
    return df


def downsample_and_shift(prices, timeframe, level='ticker', d_shift=30):
    if timeframe == '5Min':
        t_max = 5
    elif timeframe == '15Min':
        t_max = 15
    elif timeframe == '6H':
        t_max = 60*6
    elif timeframe == '12H':
        t_max = 60*12
    elif timeframe in ['60Min', 'H']:
        t_max = 60
    elif timeframe in ['D', '24H']:
        t_max = 24 * 60
    elif timeframe == 'M':
        t_max = 28 * 24 * 60
    else:
        raise NotImplementedError
    dataset = []
    t = 0
    while t < t_max:
        offset = f'{t}Min'
        print(f"offset {offset}")
        price = pd.concat([prices['open'].unstack(level=level).resample(timeframe, offset=offset).first(),
                           prices['high'].unstack(level=level).resample(timeframe, offset=offset).max(),
                           prices['low'].unstack(level=level).resample(timeframe, offset=offset).min(),
                           prices['close'].unstack(level=level).resample(timeframe, offset=offset).last(),
                           prices['volume'].unstack(level=level).resample(timeframe, offset=offset).sum(min_count=1), ],
                          keys=['open', 'high', 'low', 'close', 'volume'], axis=1).stack(level=level, dropna=True)
        # if t > 0:
        #    price.reset_index(level='ticker', inplace=True)
        #    price.ticker += f'-{t}Min'
        #    price.set_index(['ticker'], append=True, inplace=True)
        price.reset_index('date', inplace=True)
        price['date'] -= pd.DateOffset(minutes=t)
        price['offset'] = t
        # TODO be flexible if there are more index columns
        price.set_index(['date', 'offset'], append=True, inplace=True)
        dataset.append(price)
        t += d_shift

    prices_df = pd.concat(dataset)
    #prices_df.reset_index('offset', drop=False, inplace=True)
    #prices_df['offset'] = prices_df['offset'].astype('category')
    #prices_df.set_index('offset', append=True, inplace=True)
    prices_df.dropna(inplace=True)

    return prices_df


def downsample(prices, timeframe, level='ticker'):
    prices = pd.concat([prices['open'].unstack(level=level).resample(timeframe).first(),
                        prices['high'].unstack(level=level).resample(timeframe).max(),
                        prices['low'].unstack(level=level).resample(timeframe).min(),
                        prices['close'].unstack(level=level).resample(timeframe).last(),
                        prices['volume'].unstack(level=level).resample(timeframe).sum(), ],
                       keys=['open', 'high', 'low', 'close', 'volume'], axis=1).stack(level=[level], dropna=True)
    prices.dropna(inplace=True)
    return prices

"""
def data_cacher(params):
    prices = None
    cryptos = None
    categories = None

    key = params_to_key(params)
    # load data from store
    file_list = glob('../data/crypto/prices_cache/' + "*.feather")

    if key+'_prices.feather' in file_list:
        prices = pd.from_feather(key)
        cryptos = pd.from_feather(key)
        prices = pd.from_feather(key)

    cryptos = store[key + '/cryptos']
    categories = store[key + '/categories']
    print("found data in cache")

    if type(prices) is type(None):
        print("load data")
        prices, cryptos, categories = data_loader(**params)

        with pd.HDFStore(params['data_store']) as store:
            store.put(key + '/prices', prices.reset_index(), format='table')
            store.put(key + '/cryptos', cryptos, format='table')
            store.put(key + '/categories', categories)

    # already done (see above)
    
    # save memory and speed up stuff by making index categorical
    # cannot be saved as h5
    #if params['drop_level']:
    #    cols = ['ticker']
    #else:
    #    cols = ['ticker', 'symbol', 'base']
    #prices.reset_index(cols, inplace=True)
    #prices[cols] = prices[cols].astype('category')
    #prices.set_index(cols, inplace=True, append=True)
    

    return prices, cryptos, categories


def params_to_key(params):
    key = ''
    for param_key, param_value in params.items():
        key += param_key + '_' + param_value
        key += '__'
    key += '_prices.feather'
    print(key)
    return key
"""

def data_loader(timeframe='60Min',
                minutes_per_base_frequency=5,  # 5 Minutes sampled input data in data path
                prices_path='../data/crypto/prices_5Min.feather',
                min_observation_years=1,
                drop_level=True,
                exclude_stablecoins=True,
                cut=True, reduce_size=True,
                downsampleshift=False,
                d_shift=30,
                load_f32=True,
                join_sector=True,
                join_sectors=False):

    # categorical index for efficiency, but have to be dropped for storage
    multiindex_prices = ['date', 'ticker', 'symbol', 'base']
    multiindex_market = ['ticker', 'symbol', 'base']

    # load data from disk
    prices = pd.read_feather(prices_path).set_index(multiindex_prices)
    crypto_marketcap = pd.read_feather('../data/crypto/crypto_marketcap.feather').set_index(multiindex_market)
    cat_df = pd.read_feather('../data/crypto/cat_df.feather').set_index(multiindex_market)

    # remove levels if not needed
    if drop_level:
        prices = clean_df(prices)
        crypto_marketcap = clean_df(crypto_marketcap)
        cat_df = clean_df(cat_df)

    """
    if timeframe == '60Min':
        min_obs = int(365 * min_observation_years * 24)
    elif timeframe == 'D':
        min_obs = int(365 * min_observation_years)
    elif timeframe == 'M':
        min_obs = int(12 * min_observation_years)
    elif timeframe == '15Min':
        min_obs = int(365 * min_observation_years * 24 * 4)
    elif timeframe == '5Min':
        min_obs = int(365 * min_observation_years * 24 * 12)
    elif timeframe == '1Min':
        min_obs = int(365 * min_observation_years * 24 * 60)
    else:
        print(f"Please implement min observations for this timeframe {timeframe}")
        raise NotImplementedError
    """

    # assume prices are 5 min Frequency
    # TODO assert
    if min_observation_years > 0:
        min_nobs = int(365 * 24 * 60 / minutes_per_base_frequency * min_observation_years)
        nobs = prices.groupby(level='ticker').size()
        keep = nobs[nobs > min_nobs].index
        prices = prices.loc[(slice(None), keep, slice(None)), :]
        num_tickers = prices.index.get_level_values('ticker').nunique()

    if cut and min_observation_years > 0:
        # cut timeframes so that all tickers are available and unstacking doesnt produce new nan
        min_number_tickers = num_tickers
        prices = prices.unstack('ticker').dropna(thresh=min_number_tickers * 5).stack('ticker')
    elif cut and min_observation_years < 0:
        raise NotImplementedError  # does not make sense?

    # exclude stablecoins via categories
    if exclude_stablecoins:
        cat_df = cat_df[cat_df['Stablecoins'] == 0]
        cat_df = cat_df[cat_df['USD Stablecoin'] == 0]

    # get only shared data
    shared = prices.index.get_level_values('ticker').intersection(crypto_marketcap.index.get_level_values('ticker')) \
        .intersection(cat_df.index.get_level_values('ticker'))

    crypto_marketcap = crypto_marketcap.loc[shared, :]
    cat_df = cat_df.loc[shared, :]
    prices = prices.loc[(slice(None), shared), :]

    assert cat_df.shape[0] == crypto_marketcap.shape[0]
    assert prices.index.get_level_values('ticker').nunique() == cat_df.shape[0]
    assert prices.unstack('date').shape[0] == crypto_marketcap.shape[0], cat_df.shape[0]
    assert crypto_marketcap.shape[0] == cat_df.shape[0]

    # compute intense operations after cutting down prices.. also ticker changes to simplify the augmented data
    if downsampleshift:
        if drop_level:
            prices = downsample_and_shift(prices, timeframe,
                                          level='ticker', d_shift=d_shift)
            prices = prices.reorder_levels(['date', 'offset', 'ticker'])
        else:
            raise NotImplementedError
            """ does not work, don't know why.. 
            prices = downsample_and_shift(prices, timeframe,
                                          level='ticker', d_shift=d_shift)
            prices.reset_index().set_index(['date', 'offset', 'ticker', 'symbol', 'base'])
            """
    else:
        # downsample prices to timeframe
        prices = downsample(prices, timeframe)

    # only allow sectors on main sector row which have a minimum count
    min_freq = 3
    freq = crypto_marketcap['sector'].value_counts()
    frequent_values = freq[freq <= min_freq].index
    crypto_marketcap.sector.cat.add_categories(['others'], inplace=True)
    crypto_marketcap.sector[crypto_marketcap['sector'].isin(frequent_values)] = 'others'

    print("\nreducing footprint\n")
    # TODO implement dollar volume rank filter
    if reduce_size:
        prices = reduce_footprint(prices, float_type='float32', except_cols=['volume'])
        crypto_marketcap = reduce_footprint(crypto_marketcap, float_type='float32')
        cat_df = reduce_footprint(cat_df, int_type='uint16')

    if join_sector:
        prices = prices.join(crypto_marketcap['sector'])
    elif join_sectors:
        prices = prices.join(cat_df)

    return prices, crypto_marketcap, cat_df


if __name__ == "__main__":
    print("start")
    params = {'timeframe': 'D',
              'minutes_per_base_frequency': 12*60,
              'prices_path': '../data/crypto/prices_12H.feather',
              'min_observation_years': 5,
              'drop_level': True,
              'exclude_stablecoins': True,
              'cut': True,
              'reduce_size': True,
              'downsampleshift': True,
              'd_shift': 12*60,
              'load_f32': True,
              'join_sector': False,
              'join_sectors': False}

    #prices, metadata, categories = data_cacher(params)

    print("load comparison")
    prices2, metadata2, categories2 = data_loader(**params)

    #prices3, metadata3, categories3 = data_cacher(params)
    """
    assert prices == prices2
    assert metadata == metadata2
    assert categories == categories2
    assert prices == prices3
    assert metadata == metadata3
    assert categories == categories3
    print("done")
    """