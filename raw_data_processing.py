import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger


""" firm raw data processing"""
def load_raw_data(path: str)-> pd.DataFrame:

    data = pd.read_excel(path, 
                         usecols = ['代號', '名稱', '年月日', '收盤價(元)'], 
                         dtype = {'代號': str, '名稱': str, '年月日': str, '收盤價(元)': np.float32})
    return data

def rename_col_and_date_format(df: pd.DataFrame)-> pd.DataFrame:

    col = {
        '代號': 'code',
        '名稱': 'firm',
        '年月日': 'date', 
        '收盤價(元)': 'close'}
    df.rename(columns = col, inplace = True)
    df['date'] = pd.to_datetime(df['date'], format = '%Y/%m/%d')
    return df

def daily_data_processing()-> None:

    """
    input file: raw_data/firm_data/*.xlsx
    output file: data/daily_data_1.csv
    """
    # Load raw data
    df_list = list()
    for path in [f'raw_data/{i}' for i in os.listdir('raw_data/firm_data')]:
        df_list.append(load_raw_data(path))

    # Rename columns and date format
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(rename_col_and_date_format, df_list)

    # Merge data
    final_df = pd.DataFrame()
    for df in result:
        final_df = pd.concat([final_df, df], ignore_index = True)
    final_df.sort_values(by = ['code', 'date'], inplace = True)
    final_df.to_csv('data/daily_data_1.csv', index = False)
    logger.info(f'preview of merged data: \n{final_df.head()}')
    logger.info('raw data merged and saved as data/daily_data_1.csv')

def check_daily_data()-> None:

    df = pd.read_csv('data/daily_data_1.csv')
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize = (12, 6))
    for code in df['code'].unique():
        temp = df[df['code'] == code]
        plt.plot(temp['date'], temp['close'], label = code)
    plt.xticks(ticks = df['date'][::252], rotation = 45)
    plt.xlabel('date')
    plt.ylabel('close price')
    plt.title('close price')
    plt.legend()
    plt.show()

""" monthly data processing """
def create_monthly_data()-> None:

    """
    create a new column 'month' and drop duplicates
    keep the last record of each month for each firm
    reduce the size of data
    input file: data/daily_data_1.csv
    output file: data/monthly_data_1.csv
    """
    df = pd.read_csv('data/daily_data_1.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

    df.drop_duplicates(subset = ['code', 'date'], keep = 'last', inplace = True)
    col = ['code', 'firm', 'date', 'year', 'month', 'close']
    df = df[col]
    df.to_csv('data/monthly_data_1.csv', index = False)
    logger.info(f'preview of monthly data_1: \n{df.head()}')
    logger.info('monthly data created and saved as data/monthly_data_1.csv')

def check_monthly_data()-> None:

    df = pd.read_csv('data/monthly_data_1.csv')
    df = df[df['code'] == 1101]
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize = (10, 6))
    plt.plot(df['date'], df['close'], label = 'close price')
    plt.xticks(ticks = df['date'][::12], rotation = 45)
    plt.xlabel('date')
    plt.ylabel('close price')
    plt.title('close price')
    plt.legend()
    plt.show()

""" risk free rate processing """
def load_risk_free_rate(path: str)-> pd.DataFrame:

    df = pd.read_excel(path)
    return df

def risk_free_rate_processing()-> None:

    """
    input file: raw_data/risk_free_rate/*.xlsx
    output file: data/risk_free_rate.csv
    """
    df = pd.DataFrame()
    path_list = os.listdir('raw_data/risk_free_rate')
    for path in path_list:
        df = pd.concat([df, load_risk_free_rate(f'raw_data/risk_free_rate/{path}')], ignore_index = True)

    df.rename(columns = {'年月日': 'date', '無風險利率': 'rf'}, inplace = True)
    df['date'] = pd.to_datetime(df['date'], format = '%Y/%m/%d')
    df['rf'] = df['rf'].astype(np.float32) / 100
    df.sort_values(by = 'date', inplace = True)
    df.to_csv('data/risk_free_rate.csv', index = False)
    logger.info(f'preview of risk free rate data: \n{df.head()}')
    logger.info('risk free rate data loaded and saved as data/risk_free_rate.csv')

def check_risk_free_rate()-> None:

    """
    check if there are missing values in the risk free rate data
    input file: data/risk_free_rate.csv
    """
    df = pd.read_csv('data/risk_free_rate.csv')
    missing = df['rf'].isnull().sum()
    if missing > 0:
        logger.warning(f'there are {missing} missing values in the risk free rate data')
    else:
        logger.info('no missing values in the risk free rate data')
    
    plt.figure(figsize = (10, 6))
    plt.plot(df['date'], df['rf'], label = 'risk free rate')
    plt.xticks(ticks = df['date'][::252], rotation = 45)
    plt.xlabel('date')
    plt.ylabel('risk free rate')
    plt.title('risk free rate')
    plt.legend()
    plt.show()

""" market raw data processing """
def load_market_data(path: str)-> pd.DataFrame:

    df = pd.read_excel(path)
    return df

def market_data_processing()-> None:

    """
    input file: raw_data/market_data/*.xlsx
                data/risk_free_rate.csv
    output file: data/market_data_1.csv
    """
    market_df = pd.DataFrame()
    path_list = os.listdir('raw_data/market_data')
    for path in path_list:
        print(path)
        df = load_market_data(f'raw_data/market_data/{path}')
        df.columns = df.iloc[0]
        df = df[1:]
        market_df = pd.concat([market_df, df], ignore_index = True)

    market_df.rename(columns = {'年月日': 'date', '收盤價(元)': 'market_index'}, inplace = True)
    market_df['date'] = pd.to_datetime(market_df['date'], format = '%Y/%m/%d')
    rf_df = pd.read_csv('data/risk_free_rate.csv')
    rf_df['date'] = pd.to_datetime(rf_df['date'])
    market_df = market_df.merge(rf_df, on = 'date', how = 'left')
    market_df.sort_values(by = 'date', inplace = True)
    market_df = market_df[['date', 'market_index', 'rf']]
    market_df.to_csv('data/market_data_1.csv', index = False)
    logger.info(f'preview of market data_1: \n{market_df.head()}')
    logger.info('market data loaded and saved as data/market_data_1.csv')

def check_market_data()-> None:

    df = pd.read_csv('data/market_data_1.csv')
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize = (12, 6))
    plt.plot(df['date'], df['market_index'], label = 'market index', color = 'green', linewidth = 1.5)
    plt.xticks(ticks = df['date'][::252], rotation = 45)
    plt.xlabel('date')
    plt.ylabel('market index')
    plt.title('market index')
    plt.legend()
    plt.show()

""" market value data processing"""
def load_market_value(path: str)-> pd.DataFrame:
    
    df = pd.read_excel(path)
    return df

def market_value__data_processing()-> None:

    """
    input file: raw_data/market_value/*.xlsx
    output file: data/market_value_1.csv
    """
    market_value_df = pd.DataFrame()
    path_list = os.listdir('raw_data/market_value_data')
    for path in path_list:
        logger.info(f'loading {path}')
        df = load_market_value(f'raw_data/market_value_data/{path}')
        market_value_df = pd.concat([market_value_df, df], ignore_index = True)

    market_value_df.rename(columns = {'代號': 'code', 
                                      '名稱': 'firm',
                                      '年月日': 'datetime', 
                                      '市值(百萬元)': 'market_value' 
                                      }, inplace = True)
    market_value_df['datetime'] = pd.to_datetime(market_value_df['datetime'], format = '%Y/%m/%d')
    market_value_df['date'] = pd.to_datetime(market_value_df['datetime'].dt.strftime('%Y-%m'))
    market_value_df.sort_values(by = ['code', 'date'], inplace = True)
    market_value_df = market_value_df[['code', 'firm', 'datetime', 'date', 'market_value']]
    market_value_df.to_csv('data/market_value_1.csv', index = False)
    logger.info(f'preview of market value data: \n{market_value_df.head()}')
    logger.info('market value data loaded and saved as data/market_value_1.csv')

def check_market_value()-> None:
    
    CODE = 1101
    df = pd.read_csv('data/market_value_1.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['code'] == CODE]
    plt.figure(figsize = (12, 6))
    plt.plot(df['date'], df['market_value'], label = 'market value', color = 'blue', linewidth = 1.5)
    plt.xticks(ticks = df['date'][::252], rotation = 45)
    plt.xlabel('date')
    plt.ylabel('market value')
    plt.title('market value')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # daily_data_processing()
    # check_daily_data()

    # create_monthly_data()
    # check_monthly_data()

    # risk_free_rate_processing()
    # check_risk_free_rate()
    
    # market_data_processing()
    # check_market_data()
    
    # market_value__data_processing()
    check_market_value()