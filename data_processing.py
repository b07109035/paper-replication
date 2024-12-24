import os
import multiprocessing as mp
import pandas as pd
import numpy as np
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

def raw_data_processing()-> None:

    """
    input file: raw_data/*.xlsx
    output file: data/merged_data.csv
    """
    # Load raw data
    df_list = list()
    for path in [f'raw_data/{i}' for i in os.listdir('raw_data')]:
        df_list.append(load_raw_data(path))

    # Rename columns and date format
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(rename_col_and_date_format, df_list)

    # Merge data
    final_df = pd.DataFrame()
    for df in result:
        final_df = pd.concat([final_df, df], ignore_index = True)
    final_df.sort_values(by = ['code', 'date'], inplace = True)
    final_df.to_csv('data/daily_data.csv', index = False)
    logger.info(f'preview of merged data: \n{final_df.head()}')
    logger.info('raw data merged and saved as data/daily_data.csv')

def create_monthly_data(df: pd.DataFrame)-> None:

    """
    create a new column 'month' and drop duplicates
    keep the last record of each month for each firm
    reduce the size of data
    input file: data/merged_data.csv
    output file: data/monthly_data.csv
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

    df.drop_duplicates(subset = ['code', 'date'], keep = 'last', inplace = True)
    col = ['code', 'firm', 'date', 'year', 'month', 'close']
    df = df[col]
    df.to_csv('data/monthly_data.csv', index = False)
    logger.info(f'preview of monthly data: \n{df.head()}')
    logger.info('monthly data created and saved as data/monthly_data.csv')

""" market raw data processing """
def load_market_data(path: str)-> pd.DataFrame:

    df = pd.read_excel(path)
    return df

def market_data_processing()-> None:

    """
    input file: market_data/*.xlsx
    output file: data/market_data.csv
    """
    market_df = pd.DataFrame()
    path_list = os.listdir('market_data')
    for path in path_list:
        print(path)
        df = load_market_data(f'market_data/{path}')
        df.columns = df.iloc[0]
        df = df[1:]
        market_df = pd.concat([market_df, df], ignore_index = True)

    market_df.rename(columns = {'年月日': 'date', '收盤價(元)': 'market_index'}, inplace = True)
    market_df['date'] = pd.to_datetime(market_df['date'], format = '%Y/%m/%d')
    market_df.to_csv('data/market_data.csv', index = False)
    logger.info(f'preview of market data: \n{market_df.head()}')
    logger.info('market data loaded and saved as data/market_data.csv')


if __name__ == '__main__':

    # raw_data_processing()
    # create_monthly_data(pd.read_csv('data/merged_data.csv'))

    # load market data
    market_data_processing()





































































