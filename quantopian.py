import yfinance as yf
import pandas as pd
from datetime import datetime
import statsmodels.api as sm

def get_pricing(symbol, start_date='1900-01-01', end_date=None, frequency='daily', fields=None):
    """
    Load a table of historical trade data.

    Parameters
    ----------
    symbol : Object convertible to Asset
        Valid input types are Asset, Integral, or basestring. In the case that
        the passed objects are strings, they are interpreted
        as ticker symbols and resolved relative to the date specified by
        symbol_reference_date.

    start_date : str or pd.Timestamp, optional
        String or Timestamp representing a start date or start intraday minute
        for the returned data. Defaults to '1900-01-01'.

    end_date : str or pd.Timestamp, optional
        String or Timestamp representing an end date or end intraday minute for
        the returned data. Defaults to now.

    frequency : {'daily', 'minute'}, optional
        Resolution of the data to be returned.
        Other frequencies are: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.

    fields : str or list, optional
        String or list drawn from {'price', 'open_price', 'high', 'low',
        'close_price', 'volume'}.  Default behavior is to return all fields.

    Returns
    -------
    pandas DataFrame/Series
        The pricing data that was requested. See note below.

    Notes
    -----
    If a string is passed for the value of `symbols` and `fields` is None or a
    list of strings, data is returned as a DataFrame with a DatetimeIndex and
    columns given by the passed fields.

    If a list of symbols is provided, and `fields` is a string, data is
    returned as a DataFrame with a DatetimeIndex and a columns given by the
    passed `symbols`.

    If both parameters are passed as strings, data is returned as a Series.
    """
    
    # if multiple symbols are requested
    if type(symbol) == list and fields:
        
        prices_df = pd.DataFrame()

        for item in symbol:
            prices_df[item] = get_pricing(item, start_date, end_date, frequency, fields)[fields]
        
        return prices_df.dropna()

    # end date defaults to today
    if not end_date:
        end_date = datetime.now()
    
    # convert frequency to Yahoo interval type
    if frequency == 'daily':
        frequency = '1d'
    elif frequency == 'minute': ## intraday cannot extend last 60 days
        frequency = '1m' ## can only download up to 7 days
    
    # download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=frequency)
    
    # process data
    df = df.rename(columns={'Open':'open_price', 'High':'high', 'Low':'low', 'Close':'close_price', 'Volume':'volume'})
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
    df['price'] = df['close_price']
    
    # only return given fields
    if fields:
        fields = ''.join(fields).split(',')
        df = df[fields]
    
    return df


def RollingOLS(y, x, window=30):
    """
    Returns the parameters of the rolling OLS fit
    with default window size 30.
    """
    
    result = pd.DataFrame(columns=('x', 'intercept'), index=y[window:].index)
    
    for i_start in range(len(y) - window):
        
        i_end = i_start + window
        
        x_new = sm.add_constant(x[i_start:i_end])
        y_new = y[i_start:i_end]
        
        model_fit = sm.OLS(y_new, x_new).fit()
        result.iloc[i_start] = list(model_fit.params)
    
    return result