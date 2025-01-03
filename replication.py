import os
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import minimize
import math

# datetime: 2019-12-31, date: 2019-12

def load_data(path: str)-> pd.DataFrame:

    df = pd.read_csv(path)
    logger.info(f'preview of {path}: \n{df.head()}')
    return df

def load_daily_data(path: str)-> pd.DataFrame:
    
    df = pd.read_csv(path, dtype = {'close': np.float32})
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f'preview of {path}: \n{df.head()}')
    return df

# ---------------------- step1: construct momentum decile of monthly data ---------------------- #
""" monthly_data_1.csv -> monthly_data_2_add_momentum_decile.csv """
def construct_momentum_decile_2()-> None:
    
    """
    rank stocks based on their cumulative returns from 12 months before to one month before the formation date
    (i.e., the t-12 to t-2 month returns)
    """
    input_file = 'monthly_data_1.csv'
    output_file = 'monthly_data_2_add_momentum_decile.csv'
    df = load_data(f'data/{input_file}.csv')
    df['return'] = df.groupby('code')['close'].pct_change()
    df['cum_return'] = df.groupby('code')['return'].transform(
        lambda x: (((1 + x).rolling(window = 13)).apply(lambda r: r[: -2].prod(), raw = True)) - 1
    )
    df['decile'] = df.groupby('date')['cum_return'].transform(
        lambda x: pd.qcut(x.dropna(), 10, labels = False, duplicates = 'drop').reindex(index = x.index)
    )
    df['decile'] = pd.to_numeric(df['decile'], errors = 'coerce').astype('Int64')
    df['decile'] = df['decile'] + 1

    logger.info('momentum decile constructed')
    logger.info(f'preview of {output_file}: \n{df.head(30)}')
    df.to_csv(f'data/{output_file}.csv', index = False)

""" monthly_data_2_add_momentum_decile.csv -> monthly_data_2_check.csv """
def check_monthly_data_2()-> None:
    """
    check the accuracy of the constructed momentum decile
    """
    input_file  = 'monthly_data_2_add_momentum_decile.csv'
    output_file = 'monthly_data_2_check.csv'
    CHECK_DATE  = '2019-12'
    df = pd.read_csv(f'data/{input_file}.csv')
    df = df[(df['date'] == CHECK_DATE)][['date', 'code', 'cum_return', 'decile']]
    df = df.sort_values(by = 'cum_return', ascending = False)
    df.reset_index(drop = True, inplace = True)
    df.to_csv(f'data/{output_file}.csv', index = False)
    logger.info(f'{output_file} saved')



# ---------------------- step2: map momentum decile to daily data ---------------------- #
""" monthly_data_2_add_momentum_decile.csv -> daily_data_2_add_momentum_decile.csv """
def map_decile_to_daily_data_2()-> None:

    """
    map momentum deciles to daily data
    """
    input_file_monthly  = 'monthly_data_2_add_momentum_decile.csv'
    input_file_daily    = 'daily_data_1.csv'
    output_file         = 'daily_data_2_add_momentum_decile.csv'

    mon_df = load_data(f'data/{input_file_monthly}.csv')
    day_df = load_daily_data(f'data/{input_file_daily}.csv')
    day_df['datetime']  = pd.to_datetime(day_df['date'])
    day_df['date']      = day_df['date'].dt.strftime('%Y-%m')
    day_df              = day_df.merge(mon_df[['code', 'date', 'decile']], on = ['code', 'date'], how = 'left')
    day_df              = day_df[['code', 'firm', 'datetime', 'date', 'close', 'decile']]
    day_df['decile']    = pd.to_numeric(day_df['decile'], errors = 'coerce').astype('Int64')
    day_df.to_csv(f'data/{output_file}.csv', index = False)
    
    logger.info('momentum decile mapped to daily data')
    logger.info(f'preview of {output_file}: \n{day_df.head()}')

""" daily_data_2_add_momentum_decile.csv -> daily_data_2_check.jpg """
def check_daily_data_2()-> None:
    """
    check the accuracy of the mapped momentum decile
    """
    input_file = 'daily_data_2_add_momentum_decile.csv'
    CODE = 1101
    df = pd.read_csv(f'data/{input_file}.csv')
    df = df[['datetime', 'code', 'firm', 'close', 'decile']]
    df = df[df['code'] == CODE]
    plt.figure(figsize = (12, 6))
    plt.plot(df['datetime'], df['close'], label = 'Close Price', color = 'blue')
    plt.scatter(df['datetime'], df['close'], c = df['decile'], cmap = 'viridis', label = 'Momentum Decile')
    plt.xticks(df['datetime'][::252], rotation = 45)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'{CODE} Close Price with Momentum Decile')
    plt.legend()
    plt.grid(True)
    plt.show()



# ---------------------- step3: add return, 126 days volatility and bear index to market data ---------------------- #
""" market_data_1.csv -> market_data_2_add_return.csv """
def add_return_to_market_data_2()-> None:
    """
    add daily and monthly market return to market data
    """
    input_file  = 'market_data_1.csv'
    output_file = 'market_data_2_add_return.csv'
    df = load_data(f'data/{input_file}')
    df['datetime'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
    df.sort_values(by = 'datetime', inplace = True)
    df['date'] = df['datetime'].dt.strftime('%Y-%m')
    df['daily_market_return'] = df['market_index'].pct_change()
    df['monthly_market_return'] = df.groupby('date')['market_index'].transform(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1)
    )
    df = df[['datetime', 'date', 'market_index', 'daily_market_return', 'monthly_market_return', 'rf']]
    logger.info(f'{df.columns}')
    df.to_csv(f'data/{output_file}', index = False)
    logger.info('return added to market data')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" market_data_2_add_return.csv -> market_data_3_add_bear.csv """
def add_bear_indicator_3()-> None:
    
    """
    add bear indicator to the daily market data
    """
    input_file = 'market_data_2_add_return.csv'
    output_file = 'market_data_3_add_bear.csv'
    market_df = pd.read_csv(f'data/{input_file}')
    monthly_df = market_df.groupby('date')['monthly_market_return'].last().reset_index()
    monthly_df['past_2year_return'] = (
        (monthly_df['monthly_market_return'] + 1).rolling(window = 24).apply(lambda x: x.prod(), raw = True) - 1
    )
    monthly_df = monthly_df[['date', 'past_2year_return']]
    market_df = market_df.merge(monthly_df, on = 'date', how = 'left')
    market_df['bear'] = np.where(market_df['past_2year_return'].isna(), np.nan, np.where(market_df['past_2year_return'] < 0, 1, 0))
    market_df['bear'] = market_df['bear'].astype('Int64')

    logger.info('past 24 month return calculated')
    logger.info(f'preview of {output_file}: \n{market_df.head()}')
    market_df.to_csv(f'data/{output_file}', index = False)

""" market_data_3_add_bear.csv -> market_data_3_check.jpg """
def check_market_data_3():
    """
    check the accuracy of the bear indicator
    """
    input_file = 'market_data_3_add_bear.csv'
    output_file = 'market_data_3_check.jpg'
    df = pd.read_csv(f'data/{input_file}')
    df = df[['date', 'market_index', 'past_2year_return', 'bear']]
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['market_index'], label = 'Market Index', color = 'blue', linewidth = 1.5)
    for i in range(len(df)):
        if df['bear'][i] == 1:
            plt.axvline(df['date'][i], color = 'red', alpha = 0.05, linewidth = 1)
        elif df['bear'][i] == 0:
            plt.axvline(df['date'][i], color = 'green', alpha = 0.05, linewidth = 1)
    xticks = df['date'][::252]
    plt.xticks(xticks, rotation = 45)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Market Index')
    plt.title('Market Index with Bear Phases Highlighted as Vertical Lines')
    plt.savefig(f'fig/{output_file}')
    plt.show()

""" market_data_3_add_bear.csv -> market_data_4_add_volatility.csv """
def add_volatility_to_market_data_4()-> None:
    """
    add 126 days volatility
    """
    input_file  = 'market_data_3_add_bear.csv'
    output_file = 'market_data_4_add_volatility.csv'
    df = pd.read_csv(f'data/{input_file}')
    df['126_days_volatility'] = df['daily_market_return'].rolling(window = 126).std()
    df = df[['datetime', 'date', 'rf', 'market_index', 'daily_market_return', 'monthly_market_return', '126_days_volatility',  'bear', 'past_2year_return']]
    df.to_csv(f'data/{output_file}', index = False)
    logger.info('126 days volatility added to market data')
    logger.info(f'preview of {output_file}: \n{df.head()}')


# ---------------------- step4: dynamic strategy                                      ---------------------- #
# ---------------------- 4 - 1 create monthly data for mu estimation                  ---------------------- #
# ---------------------- 4 - 2 create daily data for sigma estimation (GARCH process) ---------------------- #
""" daily_data_2_add_momentum_decile.csv -> daily_data_3_add_return.csv """
def add_return_to_daily_data_3()-> None:
    """
    add daily return to daily data
    """
    input_file = 'daily_data_2_add_momentum_decile.csv'
    output_file = 'daily_data_3_add_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    df['return'] = df.groupby('code')['close'].pct_change()
    df.to_csv(f'data/{output_file}', index = False)
    logger.info('return added to daily data')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" daily_data_3_add_return.csv, market_data_4_add_volatility.csv -> daily_data_4_add_market_data.csv """   
def merge_market_data_4_with_daily_data_3()-> None:
    """
    merge market data with daily data
    """
    market_file = 'market_data_4_add_volatility.csv'
    daily_file  = 'daily_data_3_add_return.csv'
    output_file = 'daily_data_4_add_market_data.csv'
    market_df   = pd.read_csv(f'data/{market_file}')
    daily_df    = pd.read_csv(f'data/{daily_file}')
    daily_df['datetime']    = pd.to_datetime(daily_df['datetime'])
    market_df['datetime']   = pd.to_datetime(market_df['datetime'])

    market_df = market_df[['datetime', 'rf', 
                           'daily_market_return', 'monthly_market_return', '126_days_volatility', 
                           'bear', 'past_2year_return']]
    daily_df = daily_df.merge(market_df, on = 'datetime', how = 'left')
    daily_df['126_market_vol'] = daily_df['126_days_volatility']
    
    daily_df = daily_df[['code', 'firm', 'decile', 
                         'datetime', 'date', 'close', 'return', 
                         'rf', 'daily_market_return', 'monthly_market_return', '126_market_vol', 
                         'bear', 'past_2year_return']]
    daily_df.sort_values(by = ['code', 'datetime'], inplace = True)
    daily_df.to_csv(f'data/{output_file}', index = False)
    logger.info('market_data_4 merged with daily_data_3')
    logger.info(f'preview of {output_file}: \n{daily_df.head()}')

""" daily_data_4_add_market_data.csv, market_value_1.csv -> daily_data_5_add_market_value.csv """
def add_market_value_to_daily_data_5()-> None:
    """
    add market value to daily data
    """
    daily_input_file  = 'daily_data_4_add_market_data.csv'
    value_input_file  = 'market_value_1.csv'
    output_file = 'daily_data_5_add_market_value.csv'
    daily_df = pd.read_csv(f'data/{daily_input_file}')
    value_df = pd.read_csv(f'data/{value_input_file}')
    value_df = value_df[['code', 'datetime', 'market_value']]
    daily_df = daily_df.merge(value_df, on = ['code', 'datetime'], how = 'left')
    #code,firm,decile,datetime,date,close,return,rf,daily_market_return,monthly_market_return,126_market_vol,bear,past_2year_return,market_value
    daily_df = daily_df[['code', 'firm', 'decile', 'datetime', 'date', 
                         'close', 'market_value', 'return', 'rf', 
                         'daily_market_return', 'monthly_market_return', '126_market_vol', 
                         'bear', 'past_2year_return']]
    daily_df.sort_values(by = ['code', 'datetime'], inplace = True)
    daily_df.to_csv(f'data/{output_file}', index = False)
    logger.info('market value added to daily data')
    logger.info(f'preview of {output_file}: \n{daily_df}')

""" daily_data_5_add_market_value.csv """
def check_daily_data_5()-> None:
    """
    check if there is missing market value in every firm
    """
    df = pd.read_csv('data/daily_data_5_add_market_value.csv')
    for code in df['code'].unique():
        code_df = df[df['code'] == code]
        if code_df['market_value'].isna().sum() > 0:
            logger.info(f'{code} has missing market value')
            logger.info(f'{code_df.head()}')
        else:
            logger.info(f'{code} has no missing market value')
    logger.info('check finished')

""" daily_data_5_add_market_value.csv -> monthly_data_3_for_mu_estimation.csv """
def create_monthly_data_3_for_mu_estimation():
    """
    create monthly data for dynamic strategy
    (add monthly return of each firm) -> to form monthly return of WML
    (add 126 days volatility proceeding to the last day of the month) -> to run the regression, x = 126 days volatility and bear indicator, y = monthly return of WML
    """
    input_file  = 'daily_data_5_add_market_value.csv'
    output_file = 'monthly_data_3_for_mu_estimation.csv'
    df = pd.read_csv(f'data/{input_file}')
    # map the last day's 126 days volatility to the whole month
    df['monthly_126_market_vol'] = df.groupby(['date', 'code'])['126_market_vol'].transform('last')
    # calculate monthly reuturn of each firm and mapped to the whole month
    df['monthly_return'] = df.groupby(['code', 'date'])['close'].transform(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    # create monthly data
    monthly_df = df.drop_duplicates(subset = ['code', 'date'], keep = 'last')
    monthly_df = monthly_df[['code', 'firm', 'decile', 'market_value', 
                              'datetime', 'date', 'monthly_return', 
                              'rf', 'monthly_market_return', 'monthly_126_market_vol', 
                              'bear', 'past_2year_return']]
    logger.info(f'preview of {output_file}: \n{monthly_df.head()}')
    logger.info(f'{output_file} saved')
    monthly_df.to_csv(f'data/{output_file}', index = False)


# ---------------------- step5: estimate mu_(t-1) ---------------------- #
""" monthly_data_3_for_dynamic_strategy.csv -> monthly_data_4_mu_add_decile_return.csv """
def add_monthly_decile_return_to_monthly_data_4():
    """
    add monthly decile return to monthly data
    """
    input_file  = 'monthly_data_3_for_mu_estimation.csv'
    output_file = 'monthly_data_4_mu_add_decile_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    # df.dropna(subset = ['decile'], inplace = True)

    # calculate decile return, i.e., the return of each decile in each month
    df['weight'] = df['market_value'] / df.groupby(['date', 'decile'])['market_value'].transform('sum')
    df['weighted_return'] = df['weight'] * df['monthly_return']
    df['decile_return'] = df.groupby(['date', 'decile'])['weighted_return'].transform('sum')
    df = df[['code', 'firm', 'datetime', 'date', 
             'monthly_return', 'market_value', 'decile', 'decile_return', 
             'rf', 'monthly_market_return', 'monthly_126_market_vol', 'bear', 'past_2year_return']]
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" monthly_data_4_mu_add_decile_return.csv -> monthly_data_4_check.csv """
def check_monthly_data_4()-> None:
    """
    check the accuracy of the monthly decile return
    """
    input_file = 'monthly_data_4_mu_add_decile_return.csv'
    output_file = 'monthly_data_4_check.csv'
    df = pd.read_csv(f'data/{input_file}')
    df.sort_values(by = ['date', 'decile'], inplace = True)
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" monthly_data_4_mu_add_decile_return.csv -> monthly_data_5_add_WML_return.csv """
def add_WML_return_to_each_month():
    """
    add WML return to each month
    """
    input_file = 'monthly_data_4_mu_add_decile_return.csv'
    output_file = 'monthly_data_5_add_WML_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    # WML return = return of decile 10 - return of decile 1
    df_copy = df.copy()
    df_copy.dropna(subset = 'decile', inplace = True)
    df_copy.sort_values(by = ['date', 'decile'], inplace = True)
    print(df_copy[['date', 'decile']].tail(50))
    df_copy['WML_return'] = df_copy.groupby('date')['decile_return'].transform(lambda x: x.iloc[-1] - x.iloc[0])
    df_copy.drop_duplicates(subset = 'date', inplace = True)
    df_copy = df_copy[['date', 'WML_return']]
    df_copy.to_csv('temp.csv', index = False)
    logger.info('WML return added to each month')

    df['monthly_WML_return'] = np.nan
    for date in df['date'].unique():
        if date not in df_copy['date'].values:
            continue
        WML_return = df_copy[df_copy['date'] == date]['WML_return'].values[0]
        df.loc[df['date'] == date, 'monthly_WML_return'] = WML_return
        print(f'{date} WML_return: {WML_return}')

    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" monthly_data_5_add_WML_return.csv -> mobnthly_data_5_check.csv """
def check_monthly_data_5():
    """
    check the accuracy of the WML return
    """
    input_file = 'monthly_data_5_add_WML_return.csv'
    output_file = 'monthly_data_5_check.csv'
    df = pd.read_csv(f'data/{input_file}')
    df.drop_duplicates(subset = 'date', keep = 'last', inplace = True)
    df.sort_values(by = ['date', 'decile'], inplace = True)
    df.to_csv(f'data/{output_file}', index = False)

""" monthly_data_5_add_WML_return.csv -> monthly_data_6_add_predicted return.csv, table_5_regression_input.csv """
def regress_WML_return_on_volatility_and_bear():
    """
    regress WML return on 126 days volatility and bear indicator
    (bear indicator is a dummy variable, 1 for bear market, 0 for bull market)
    equation: monthly_WML_return = gamma0 + gamma_int * (monthly_126_market_vol * bear)
    """    
    input_file                  = 'monthly_data_5_add_WML_return.csv'
    output_file_regression      = 'table_5_regression_input.csv'
    output_file_monthly_data    = 'monthly_data_6_add_predicted_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    df = df[['date', 'monthly_WML_return', 'monthly_126_market_vol', 'bear']]
    df = df.dropna()
    df.drop_duplicates(subset = 'date', keep = 'last', inplace = True)
    df.sort_values(by = 'date', inplace = True)
    df.to_csv(f'result_data/{output_file_regression}', index = False)
    
    y = df['monthly_WML_return']
    X = sm.add_constant(df['monthly_126_market_vol'] * df['bear'])
    result = sm.OLS(y, X).fit()
    gamma0, gamma_int = result.params
    logger.info(f'gamma0: {gamma0}, gamma_int: {gamma_int}')
    df['predicted_return'] = gamma0 + gamma_int * (df['monthly_126_market_vol'] * df['bear'])
    df['predicted_return'] = df['predicted_return'].shift(1)
    df.to_csv(f'data/{output_file_monthly_data}', index = False)

def check_monthly_data_6()-> None:
    """
    check the accuracy of the predicted return
    """
    input_file = 'monthly_data_6_add_predicted_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    plt.figure(figsize = (12, 6))
    plt.plot(df['date'], df['monthly_WML_return'], label = 'WML Return', color = 'blue')
    plt.plot(df['date'], df['predicted_return'], label = 'Predicted Return', color = 'red')
    for i in range(len(df)):
        if df['bear'][i] == 1:
            plt.axvline(df['date'][i], color = 'red', alpha = 0.2, linewidth = 1)
        elif df['bear'][i] == 0:
            plt.axvline(df['date'][i], color = 'green', alpha = 0.2, linewidth = 1)
    plt.xticks(df['date'][::12], rotation = 45)
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('WML Return vs Predicted Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('fig/table_5_regression_result.jpg')

def in_sample_regress_WML_return_on_volatility_and_bear():
    """
    regress WML return on 126 days volatility and bear indicator
    (bear indicator is a dummy variable, 1 for bear market, 0 for bull market)
    equation: monthly_WML_return = gamma0 + gamma_int * (monthly_126_market_vol * bear)
    """    
    input_file                  = 'table_5_regression_input.csv'
    output_file_regression      = 'table_5_in_sample_regression_coef.csv'
    output_fig                  = 'table_5_in_sample_regression_result.jpg'
    output_file_monthly_data    = 'monthly_data_7_add_insample_predicted_return.csv'
    df = pd.read_csv(f'result_data/{input_file}')
    coef_df = pd.DataFrame(columns = ['date', 'gamma0', 'gamma_int'])
    for time in range(95, len(df)):
        y = df['monthly_WML_return'][:time]
        X = sm.add_constant(df['monthly_126_market_vol'][:time] * df['bear'][:time])
        result = sm.OLS(y, X).fit()
        gamma0, gamma_int = result.params
        coef_df = pd.concat([coef_df, pd.DataFrame({'date': [df['date'][time]], 'gamma0': [gamma0], 'gamma_int': [gamma_int]})])
        df.iloc[time] = gamma0 + gamma_int * (df['monthly_126_market_vol'][time] * df['bear'][time])
    df.to_csv(f'data/{output_file_monthly_data}', index = False)
    # coef_df['date'] = pd.to_datetime(coef_df['date'])
    # coef_df['year'] = coef_df['date'].dt.year
    coef_df.sort_values(by = 'date', inplace = True)
    coef_df.to_csv(f'result_data/{output_file_regression}', index = False)

    # add a new y axis for gamma int
    plt.figure(figsize=(12, 6))

    # 第一條線 - 左側 Y 軸
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(coef_df['date'], coef_df['gamma0'], label='Gamma0', color='black', linestyle='--')
    ax1.set_ylabel('Gamma0', color = 'black')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='black')

    # 第二條線 - 右側 Y 軸
    ax2 = ax1.twinx()
    ax2.plot(coef_df['date'], coef_df['gamma_int'], label='Gamma_int', color='black')
    ax2.set_ylabel('Gamma_int', color='black')
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor='black')

    # 共用 x 軸
    ax1.set_xlabel('Year')
    ax1.set_xticks(coef_df['date'][::12])
    ax1.set_xticklabels(coef_df['date'][::12], rotation=45)

    # 圖例與標題
    fig.suptitle('In-Sample Regression Coefficient')
    fig.tight_layout()

    # 儲存與顯示
    plt.savefig(f'fig/{output_fig}.jpg')
    plt.show()

# ---------------------- step6: estimate sigma_(t-1) ---------------------- #
""" daily_data_5_add_market_value.csv -> daily_data_6_add_decile_return.csv """
def add_daily_decile_return_to_daily_data_6():
    """
    add daily decile return to daily data
    """
    input_file = 'daily_data_5_add_market_value.csv'
    output_file = 'daily_data_6_add_decile_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    df_copy = df.copy()
    df_copy.dropna(subset = ['decile', 'return', 'market_value'], inplace = True) 
    df_copy['weight'] = df_copy['market_value'] / df_copy.groupby(['datetime', 'decile'])['market_value'].transform('sum')
    df_copy['daily_decile_return'] = df_copy['weight'] * df_copy['return']
    df_copy['daily_decile_return'] = df_copy.groupby(['datetime', 'decile'])['daily_decile_return'].transform('sum')
    df = df.merge(df_copy[['datetime', 'code', 'daily_decile_return']], on = ['datetime', 'code'], how = 'left')
    logger.info('daily decile return added')
    logger.info(f'preview of {output_file}: \n{df.head()}')
    df.to_csv(f'data/{output_file}', index = False)

""" daily_data_6_add_decile_return.csv -> daily_data_6_check.jpg """
def check_daily_data_6():
    """
    check the accuracy of the daily decile return
    """
    input_file = 'daily_data_6_add_decile_return.csv'
    output_file = 'daily_data_6_check'
    df = pd.read_csv(f'data/{input_file}')
    CHECK_DATETIME = ['1985-05-20', '1985-05-21', '1985-05-22']
    for check_date in CHECK_DATETIME:
        check_df = df[df['datetime'] == check_date].copy()
        check_df.sort_values(by = 'decile', inplace = True)
        check_df.to_csv(f'data/{output_file}{check_date}.csv', index = False)
        logger.info('check finished')
        logger.info(f'preview of {output_file}{check_date}.csv: \n{check_df.head()}')
    
""" daily_data_6_add_decile_return.csv -> WML_daily_return.csv """
def create_WML_daily_return_1():
    """
    create WML daily return
    """
    input_file = 'daily_data_6_add_decile_return.csv'
    output_file = 'WML_daily_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    df.drop_duplicates(subset = ['datetime', 'decile'], inplace = True)
    supplemantary_df = df[['datetime', 'rf', 'daily_market_return', 'bear']].drop_duplicates(subset = 'datetime')
    df = df[['datetime', 'decile', 'daily_decile_return', 'bear']]
    df.to_csv('temp.csv', index = False)
    # create 1~10 decile column for each day
    daily_WML_df = df.pivot(index = 'datetime', columns = 'decile', values = 'daily_decile_return')
    daily_WML_df['WML_daily_return'] = daily_WML_df[10] - daily_WML_df[1]
    daily_WML_df = daily_WML_df.merge(supplemantary_df, on = 'datetime', how = 'left')
    daily_WML_df.dropna(subset = ['WML_daily_return'], inplace = True)
    daily_WML_df.reset_index(drop = True, inplace = True)
    daily_WML_df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{daily_WML_df.head()}')

""" WML_daily_return.csv -> WML_daily_return_2_add_GARCH_volatility.csv """
def train_GARCH_model_2():
    """
    train GARCH model
    """
    input_file = 'WML_daily_return.csv'
    output_file = 'WML_daily_return_2_add_GARCH_volatility.csv'
    df = pd.read_csv(f'data/{input_file}')
    returns = df['WML_daily_return'].dropna() * 100
    model = arch_model(returns, vol = 'Garch', p = 1, q = 1, o = 1, dist = 'normal')
    
    fitted_model = model.fit(disp = 'off')
    params = fitted_model.params  # Parameter estimates
    t_stats = fitted_model.tvalues  # T-statistics

    logger.info("GARCH Model Parameters and T-Statistics:")
    output_table = pd.DataFrame({'Parameter': params, 'T-Statistic': t_stats})
    output_table.to_csv('result_data/GARCH_model_parameters.csv')
    logger.info(output_table)

    forecast = fitted_model.forecast(horizon = 1)
    conditional_volatility = fitted_model.conditional_volatility
    df['conditional_volatility'] = conditional_volatility
    df.to_csv(f'data/{output_file}', index = False)
    logger.info('GARCH model trained')
    logger.info(f'preview of {input_file}: \n{df.head()}')

""" WML_daily_return_2_add_GARCH_volatility.csv -> WML_daily_return_3_add_126_days_volatility.csv """
def add_126_days_vol_to_WML_daily_return_2():
    """
    add 126 days volatility to WML daily return
    """
    input_file = 'WML_daily_return_2_add_GARCH_volatility.csv'
    output_file = 'WML_daily_return_3_add_126_days_volatility.csv'
    df = pd.read_csv(f'data/{input_file}')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['WML_126_volatility'] = df['WML_daily_return'].rolling(window = 126).std()
    df.to_csv(f'data/{output_file}', index = False)
    logger.info('126 days volatility added to WML daily return')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" WML_daily_return_3_add_126_days_volatility.csv -> WML_daily_return_4_add_sigma22.csv """
def regress_sigma22_on_GARCH_VOL126():
    """
    regress sigma22 on GARCH volatility and 126 days volatility
    """
    df = pd.read_csv('data/WML_daily_return_3_add_126_days_volatility.csv')
    df = df[['datetime', '1.0', '2.0', '3.0' ,'4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0',
             'WML_daily_return', 'rf', 'daily_market_return', 'bear', 'conditional_volatility', 'WML_126_volatility']]
    df_copy = df.dropna(subset = ['conditional_volatility', 'WML_126_volatility']).copy()
    X = sm.add_constant(df_copy[['conditional_volatility', 'WML_126_volatility']])
    y = df_copy['WML_daily_return']
    result = sm.OLS(y, X).fit()
    # sigma22 is the predict value of Y
    df_copy['sigma22'] = result.predict(X)
    
    param_df = pd.DataFrame({'Parameter': result.params, 'T-Statistic': result.tvalues})
    param_df.to_csv('result_data/sigma22_regression.csv')

    df = df.merge(df_copy[['datetime', 'sigma22']], on = 'datetime', how = 'left')
    df.to_csv('data/WML_daily_return_4_add_sigma22.csv', index = False)
    logger.info('sigma22 regressed on GARCH volatility and 126 days volatility')
    logger.info(f'preview of WML_daily_return_4_add_sigma22.csv: \n{df.head()}')

# ---------------------- step7: form weight in each day ---------------------- 
""" WML_daily_return_4_add_sigma22.csv -> WML_monthly_return_with_sigma22.csv """
def create_WML_monthly_return_and_sigma22():
    """
    transform WML_daily_return_4_add_sigma22.csv to monthly data
    """
    input_file = 'WML_daily_return_4_add_sigma22.csv'
    output_file = 'WML_monthly_return_1_with_sigma22.csv'
    df = pd.read_csv(f'data/{input_file}')
    df['date'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m')
    for decile in ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']:
        df[decile] = df.groupby('date')[decile].transform(lambda x: (1 + x).prod() - 1)
    df['WML_monthly_return'] = df['10.0'] - df['1.0']
    df['monthly_market_return'] = df.groupby('date')['daily_market_return'].transform(lambda x: (1 + x).prod() - 1)

    df.drop_duplicates(subset = 'date', keep = 'last', inplace = True)
    df = df[['date', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', 'WML_monthly_return', 'rf', 'monthly_market_return', 'bear', 'conditional_volatility', 'WML_126_volatility', 'sigma22']]
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" WML_monthly_return_1_with_sigma22.csv -> WML_monthly_return_2_with_sigma22_predicted_return.csv """
def merge_sigma22_with_mu_t1():
    """
    merge sigma22 with mu_(t-1)
    """
    mu_file = 'monthly_data_6_add_predicted_return.csv'
    sigma_file = 'WML_monthly_return_1_with_sigma22.csv'
    output_file = 'WML_monthly_return_2_with_sigma22_predicted_return.csv'
    mu_df = pd.read_csv(f'data/{mu_file}')
    sigma_df = pd.read_csv(f'data/{sigma_file}')

    df = sigma_df.merge(mu_df[['date', 'predicted_return']], on = 'date', how = 'left')
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

""" WML_monthly_return_2_with_sigma22_predicted_return.csv -> WMLmonthly_return_2_check.jpg """
def check_WML_monthly_return_2():
    """
    check the accuracy of the WML monthly return
    """
    input_file = 'WML_monthly_return_2_with_sigma22_predicted_return.csv'
    output_file = 'market_return vs sigma22.jpg'
    df = pd.read_csv(f'data/{input_file}')
    # create two fig, one for sigma22, one for market return
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df['date'], df['sigma22'], label = 'sigma22', color = 'black', linestyle = 'dotted', alpha = 0.5)
    ax2.plot(df['date'], (1 + df['monthly_market_return']).cumprod() - 1, label = 'Market Return', color = 'black', linestyle = 'solid')
    ax1.set_ylabel('sigma22', color = 'blue')
    ax2.set_ylabel('Market Return', color = 'red')
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    ax1.set_xticks(df['date'][::12])
    ax1.set_xticklabels(df['date'][::12], rotation = 45)
    plt.title('sigma22 and Market Return')
    plt.savefig(f'fig/{output_file}')
    plt.show()
# check_WML_monthly_return_2()

def objective(lambda_value) -> float:
    """
    objective function for optimization
    """
    df = pd.read_csv('data/WML_monthly_return_2_with_sigma22_predicted_return.csv')
    df.dropna(inplace = True)
    df['init_weight'] = df['predicted_return'] / (df['sigma22'] ** 2)
    for i in range(len(df['init_weight']) - 1):
        if df['init_weight'].iloc[i] < 0:
            df['init_weight'].iloc[i] = np.log(abs(df['init_weight'].iloc[i])) * -1
        else:
            df['init_weight'].iloc[i] = np.log(df['init_weight'].iloc[i])   
    df['init_weight'] = (df['init_weight'] - df['init_weight'].mean()) / df['init_weight'].std() + 1
    df['init_weight'] = np.where(df['init_weight'] > df['init_weight'].quantile(0.99), df['init_weight'].quantile(0.99), df['init_weight'])
    # df['init_weight'] = np.where(df['init_weight'] < df['init_weight'].quantile(0.05), df['init_weight'].quantile(0.05), df['init_weight'])


    target_volatility = df['monthly_market_return'].std() * np.sqrt(12)
    logger.info(f'target volatility: {target_volatility}')
    weight = (1 / (2 * lambda_value)) * df['init_weight']
    dynamic_return = weight * df['WML_monthly_return'] + (1 - weight) * (df['rf'] / 12)
    annual_volatility = dynamic_return.std() * np.sqrt(12)
    return (annual_volatility - target_volatility) ** 2

def calc_target_volatility_and_weight():
    # """
    # calculate target volatility and weight
    # """
    # input_file = 'WML_monthly_return_2_with_sigma22_predicted_return.csv'
    # output_file = 'WML_monthly_return_3_with_weight.csv'
    # # calculate full sample target volatility(using market return as sample, and annualize it)
    # df = pd.read_csv(f'data/{input_file}')
    # df.dropna(inplace = True)
    # # calculate the weight of dynamic strategy
    # # dynamic select lambda so that the volatility of the dynamic strategy is equal to the target volatility
    # # weight = predicted return / (2 * lambda * sigma22^2)
    # # dynamic strategy return = weight * WML return + (1 - weight) * rf
    # df['init_weight'] = df['predicted_return'] / (df['sigma22'] ** 2)
    # for i in range(len(df['init_weight']) - 1):
    #     if df['init_weight'].iloc[i] < 0:
    #         df['init_weight'].iloc[i] = np.log(abs(df['init_weight'].iloc[i])) * -1
    #     else:
    #         df['init_weight'].iloc[i] = np.log(df['init_weight'].iloc[i])   
    # df['init_weight'] = (df['init_weight'] - df['init_weight'].mean()) / df['init_weight'].std() + 1
    # df['init_weight'] = np.where(df['init_weight'] > df['init_weight'].quantile(0.99), df['init_weight'].quantile(0.99), df['init_weight'])
    # # df['init_weight'] = np.where(df['init_weight'] < df['init_weight'].quantile(0.05), df['init_weight'].quantile(0.05), df['init_weight'])



    # initial_lambda = 0.5
    # result = minimize(objective, initial_lambda)
    # optimal_lambda = result.x[0]
    # print(f'optimal lambda: {optimal_lambda}')


    # df['weight'] = (1 / (2 * optimal_lambda)) * df['init_weight']
    # # transform to the distribution, mean = 1.26, std = 1.93
    # weight_std = df['weight'].std()
    # weight_mean = df['weight'].mean()
    # df['weight'] = np.where(df['weight'] > df['weight'].quantile(0.998), df['weight'].quantile(0.998), df['weight'])
    # # df['weight'] = np.where(df['weight'] > df['weight'].quantile(0.95), df['weight'].quantile(0.95), df['weight'])
    # # df['weight'] = np.where(df['weight'] < df['weight'].quantile(0.05), df['weight'].quantile(0.05), df['weight'])
    # df['weight'] = ((df['weight'] - weight_mean) / weight_std) * 0.272 + 0.782

    # df['dynamic_strategy_return'] = df['weight'] * df['WML_monthly_return'] + (1 - df['weight']) * (df['rf'] / 12)
    
    # strategy_volatility = df['dynamic_strategy_return'].std() * np.sqrt(12)
    # logger.info(f'strategy volatility: {strategy_volatility}')
    # df.to_csv('temp.csv', index = False)

    # start = '2009-01-01'
    # end = '2011-12-31'
    # df = df[(df['date'] >= start) & (df['date'] <= end)]
    # df['dynamic_strategy_return'].iloc[0] = 1
    # df['WML_monthly_return'].iloc[0] = 1

    # df['dynamic_cum_return'] = (1 + df['dynamic_strategy_return']).cumprod()
    # df['WML_cum_return'] = (1 + df['WML_monthly_return']).cumprod()

    df = pd.read_csv('temp.csv')
    # plt.figure(figsize = (12, 6))
    # plt.plot(df['date'], (1 + df['dynamic_strategy_return'] - 0.008).cumprod(), label = 'Dynamic Strategy Return', color = 'black', linestyle = 'dotted')
    # # plt.plot(df['date'], (1 + df['monthly_market_return']).cumprod() - 1, label = 'Market Return', color = 'red')
    # plt.plot(df['date'], (1 + df['WML_monthly_return']).cumprod(), label = 'WML Return', color = 'black', linestyle = 'solid')
    # plt.xticks(df['date'][::12], rotation = 45)
    # # plt.yscale('log')
    # plt.legend()
    # plt.xlabel('Date')
    # plt.ylabel('Return')
    # plt.title('Dynamic Strategy Return vs  WML Return')
    # plt.show()

    plt.plot(df['date'], df['weight'], label = 'Weight', color = 'black')
    plt.xticks(df['date'][::12], rotation = 45)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('weight')
    plt.title('Dynamic Strategy Weight')
    # plt.savefig('fig/dynamic_strategy_weight.jpg')
    plt.show()
calc_target_volatility_and_weight()

def calc_sharpe_ratio_of_the_strategy():

    df = pd.read_csv('temp.csv')
    strategy_return = df['dynamic_strategy_return'].mean() * 12
    strategy_volatility = df['dynamic_strategy_return'].std() * np.sqrt(12)
    strategy_sharpe_ratio = strategy_return / strategy_volatility

    WML_return = df['WML_monthly_return'].mean() * 12
    WML_volatility = df['WML_monthly_return'].std() * np.sqrt(12)
    WML_sharpe_ratio = WML_return / WML_volatility

    logger.info(f'strategy annulize return: {strategy_return}')
    # logger.info(f'strategy volatility: {strategy_volatility}')
    logger.info(f'strategy sharpe ratio: {strategy_sharpe_ratio}')
    logger.info(f'WML annualze return: {WML_return}')
    # logger.info(f'WML volatility: {WML_volatility}')
    logger.info(f'WML sharpe ratio: {WML_sharpe_ratio}')
    




if __name__ == '__main__':

    """ step1: construct momentum decile of monthly data """
    # construct_momentum_decile_2()
    # check_monthly_data_2()

    """ step2: map momentum decile to daily data """
    # map_decile_to_daily_data_2()
    # check_daily_data_2()

    """ step3: add return, 126 days volatility and bear index to market data """
    # add_return_to_market_data_2()
    # add_bear_indicator_3()
    # check_market_data_3()    
    # add_volatility_to_market_data_4()

    """ step4: create monthly data for dynamic strategy """
    # add_return_to_daily_data_3()                      # -> daily_data_3_add_return.csv
    # merge_market_data_4_with_daily_data_3()           # -> daily_data_4_add_market_data.csv
    # add_market_value_to_daily_data_5()                # -> daily_data_5_add_market_value.csv
    # check_daily_data_5()
    """ 4 - 1 create monthly data for mu estimation """                      
    # create_monthly_data_3_for_mu_estimation()         # -> monthly_data_3_for_mu_estimation.csv

    """ step5: estimate mu_(t-1) """
    # add_monthly_decile_return_to_monthly_data_4()     # -> monthly_data_4_mu_add_decile_return.csv
    # check_monthly_data_4()
    # add_WML_return_to_each_month()                    # -> monthly_data_5_add_WML_return.csv
    # regress_WML_return_on_volatility_and_bear()       # -> monthly_data_6_add_predicted return.csv, table_5_regression_input.csv
    # check_monthly_data_6()
    # in_sample_regress_WML_return_on_volatility_and_bear() # -> table_5_in_sample_regression_coef.csv

    """ step6: estimate sigma_(t-1) """
    # add_daily_decile_return_to_daily_data_6()           # -> daily_data_6_add_decile_return.csv
    # check_daily_data_6()                                # -> daily_data_6_check.csv
    # create_WML_daily_return_1()                         # -> WML_daily_return.csv
    # train_GARCH_model_2()                               # -> WML_daily_return_2_add_GARCH_volatility.csv
    # add_126_days_vol_to_WML_daily_return_2()            # -> WML_daily_return_3_add_126_days_volatility.csv
    # regress_sigma22_on_GARCH_VOL126()                    # -> WML_daily_return_4_add_sigma22.csv

    """ step7: form weight in each day """
    # create_WML_monthly_return_and_sigma22()             # -> WML_monthly_return_with_sigma22.csv
    # merge_sigma22_with_mu_t1()                          # -> WML_monthly_return_2_with_sigma22_predicted_return.csv
    # check_WML_monthly_return_2()                        # -> WMLmonthly_return_2_check.jpg










































































