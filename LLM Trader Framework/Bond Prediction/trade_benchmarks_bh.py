import numpy as np
import pandas as pd
import clean_input_df as input_
import datetime as dt


def cap_df( df,date_start, date_end, ):
    """
    function to custom range a df with df input 
    """
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    
    # Filter dataframe between dates (inclusive)
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Sort by date (ascending - oldest first)
    filtered_df = filtered_df.sort_index()
    
    return filtered_df


def buy_and_hold_df(df, start_date, end_date, risk_free_rate=0.0533):
    """
    Function to calculate buy and hold strategy returns and key metrics.
    """
    df = cap_df(df, start_date, end_date)
    
    # Basic Calculations: portfolio value and daily returns
    df['PV'] = df['Last Px'] / df['Last Px'].iloc[0] # Portfolio Value starting at 1
    df['Daily Return'] = df['PV'].pct_change().dropna()

    
    #Total Return
    total_return = df['PV'].iloc[-1] - 1

    #Annualized Return (ARR) - Compounded
    num_days = (df.index[-1] - df.index[0]).days

    annualized_return = ((1 + total_return) ** (365 / num_days)) - 1
        
    # Annualised Volatility
    annualized_volatility = df['Daily Return'].std() * np.sqrt(252)

    # 4. Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    # 5. Maximum Drawdown
    peak_pv = df['PV'].cummax()
    drawdowns = (peak_pv - df['PV']) / peak_pv
    max_drawdown = drawdowns.max()

    # Create a Series with the final stats
    stats = pd.Series({
        'Total Return': total_return,
        'ARR': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }, name=df.index[-1]) # Use the end date as the name of the Series
    
    return stats


print("40 year- buy and hold  ")
df_50 = input_.STRIPS_to_df(0)
df_50_stats = buy_and_hold_df(df_50, '2024-09-04', '2024-11-04')
print(df_50_stats)

print("50 year- buy and hold  ")
df_40 = input_.STRIPS_to_df(1)
df_40_stats = buy_and_hold_df(df_40, '2024-09-04', '2024-11-04')
print(df_40_stats)

print("30 year- buy and hold  ")
df_30 = input_.STRIPS_to_df(2)
df_30_stats = buy_and_hold_df(df_30, '2024-09-04', '2024-11-04')
print(df_30_stats)

df_buy_and_hold = pd.DataFrame([df_30_stats, df_40_stats, df_50_stats])

print(df_buy_and_hold)
print()

print("30 year- buy and hold longer term ")
df_30 = input_.STRIPS_to_df(2)
df_30_stats = buy_and_hold_df(df_30, '2024-06-04', '2024-11-04')
print(df_30_stats)


