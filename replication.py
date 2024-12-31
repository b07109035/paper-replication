import os
from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    daily_df.to_csv(f'data/{output_file}', index = False)
    #code,firm,decile,datetime,date,close,return,rf,daily_market_return,monthly_market_return,126_market_vol,bear,past_2year_return,market_value
    daily_df = daily_df[['code', 'firm', 'decile', 'datetime', 'date', 
                         'close', 'market_value', 'return', 'rf', 
                         'daily_market_return', 'monthly_market_return', '126_market_vol', 
                         'bear', 'past_2year_return']]
    logger.info('market value added to daily data')
    logger.info(f'preview of {output_file}: \n{daily_df.head()}')

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
    df.dropna(subset = ['decile'], inplace = True)

    # calculate decile return, i.e., the return of each decile in each month
    for decile in df['decile'].unique():
        for date in df['date'].unique():
            decile_date_df = df[(df['decile'] == decile) & (df['date'] == date)].copy()
            decile_date_df['decile_return'] = (
                (decile_date_df['monthly_return'] * decile_date_df['market_value']).sum() 
                / decile_date_df['market_value'].sum()
            )
            df.loc[(df['decile'] == decile) & (df['date'] == date), 'decile_return'] = decile_date_df['decile_return'].values[0]
        logger.info(f'decile {decile} return calculated')
    df.sort_values(by = ['date', 'decile'], inplace = True)
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

def check_monthly_data_4()-> None:
    """
    check the accuracy of the monthly decile return
    """
    input_file = 'monthly_data_4_mu_add_decile_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    for date in df['date'].unique():
        for decile in df['decile'].unique():
            decile_date_df = df[(df['decile'] == decile) & (df['date'] == date)].copy()
            weighted_return = (
                (decile_date_df['monthly_return'] * decile_date_df['market_value']).sum() / 
                decile_date_df['market_value'].sum()
            )

            logger.info(f'decile: {decile}, date: {date}')
            print(weighted_return, decile_date_df['decile_return'].values[0])
            if date == '1983-01':
                break
        break

""" monthly_data_4_mu_add_decile_return.csv -> monthly_data_5_add_WML_return.csv """
def add_WML_return_to_each_month():
    """
    add WML return to each month
    """
    input_file = 'monthly_data_4_mu_add_decile_return.csv'
    output_file = 'monthly_data_5_add_WML_return.csv'
    df = pd.read_csv(f'data/{input_file}')
    df['monthly_WML_return'] = df.groupby('date')['decile_return'].transform(lambda x: x.iloc[-1] - x.iloc[0])
    df.to_csv(f'data/{output_file}', index = False)
    logger.info(f'{output_file} saved')
    logger.info(f'preview of {output_file}: \n{df.head()}')

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
    df = pd.read_csv(f'result_data/{input_file}')
    coef_df = pd.DataFrame(columns = ['date', 'gamma0', 'gamma_int'])
    for time in range(95, len(df)):
        y = df['monthly_WML_return'][:time]
        X = sm.add_constant(df['monthly_126_market_vol'][:time] * df['bear'][:time])
        result = sm.OLS(y, X).fit()
        gamma0, gamma_int = result.params
        coef_df = pd.concat([coef_df, pd.DataFrame({'date': [df['date'][time]], 'gamma0': [gamma0], 'gamma_int': [gamma_int]})])

    coef_df['date'] = pd.to_datetime(coef_df['date'])
    coef_df['year'] = coef_df['date'].dt.year
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
    in_sample_regress_WML_return_on_volatility_and_bear() # -> table_5_in_sample_regression_coef.csv



















































































