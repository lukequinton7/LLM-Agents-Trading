import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy import stats




def cap_df(df, date_start, date_end):
    """
    manually select date range for custom df
    """
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    filtered_df = filtered_df.sort_index()
    return filtered_df

def performance_stats(trade_df, start_date, end_date, risk_free_rate=0.0533):
    """
    compute the perf metrics
    """
    df = cap_df(trade_df, start_date, end_date)

    if len(df) < 2:
        return None

    df['price_pct_change'] = df['price'].pct_change()
    df['position_lagged'] = df['new_position'].shift(1)
    df['daily_strategy_return'] = df['position_lagged'] * df['price_pct_change']
    df['pv'] = (1 + df['daily_strategy_return']).cumprod().fillna(1.0)

    total_return = df['pv'].iloc[-1] - 1

    num_days = (df.index[-1] - df.index[0]).days

    annualized_return = ((1 + total_return) ** (365 / num_days)) - 1
        
    annualized_volatility = df['daily_strategy_return'].std() * np.sqrt(252)

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    peak_pv = df['pv'].cummax()
    drawdowns = (peak_pv - df['pv']) / peak_pv
    max_drawdown = drawdowns.max()

    stats = pd.Series({
        'Total Return': total_return,
        'ARR': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }, name=df.index[-1].date())
    
    return stats





def aggregate_performance_stats(stats_list, confidence_level=0.95):
    """
    Computes means and confidence intervals for MULTIPLE performance statistics.
    """
    
    # Convert list of Series to DataFrame
    df = pd.DataFrame(stats_list)
    
    # Calculate alpha for confidence intervals
    alpha = 1 - confidence_level
    
    results = {}
    
    for metric in df.columns:
        values = df[metric].dropna()
        n = len(values)
        
        if n == 0:
            results[metric] = {'Mean': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan}
            continue
        
        mean_val = values.mean()
        
        if n == 1:
            # Single observation - no CI calculation possible
            results[metric] = {'Mean': mean_val, 'CI_Lower': np.nan, 'CI_Upper': np.nan}
        else:
            # Calculate 95% confidence interval using t-distribution
            std_err = stats.sem(values)  # Standard error of the mean
            t_critical = stats.t.ppf(1 - alpha/2, n - 1)  # t-critical value
            margin_error = t_critical * std_err
            
            ci_lower = mean_val - margin_error
            ci_upper = mean_val + margin_error
            
            results[metric] = {
                'Mean': mean_val, 
                'CI_Lower': ci_lower, 
                'CI_Upper': ci_upper
            }
    
    # Convert to DataFrame for clean output
    result_df = pd.DataFrame(results).T
    
    # Add sample size column
    result_df['N'] = len(stats_list)
    
    return result_df


#llm trade agent 1 strategies
df_30_1 = pd.read_csv(
    'bond_trades_simple_SP_0_05_15_30_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_40_1 = pd.read_csv(
    'bond_trades_simple_SP_0_05_15_40_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_1 = pd.read_csv(
    'bond_trades_simple_SP_0_05_15_50_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 



#llm trade agent 2 strategies
df_30_2 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_40_2 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_40_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_2 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 



#llm trade agent 2 strategies - ablation - no text
df_30_2_no_text = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro_no_text.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 


#llm trade agent 2 strategies - ablation - no numeric macro
df_30_2_no_numeric = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro_no_numeric.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 


#llm trade agent 2 strategies - ablation - no arima
df_30_2_no_arima = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro_no_arima.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 






#best results- statistically significant

df_50_2_trial_1 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro_TRIAL_1.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_2_trial_2 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro_TRIAL_2.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_2_trial_3 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro_TRIAL_3.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 


df_50_2_trial_4 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro_TRIAL_4.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_2_trial_5 = pd.read_csv(
    'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro_TRIAL_5.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 













#arima rule based strategy
df_30_arima = pd.read_csv(
    'rule_based_arima_SP_0_05_15_30_short.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
)

df_40_arima = pd.read_csv(
    'rule_based_arima_SP_0_05_15_40_short.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 

df_50_arima = pd.read_csv(
    'rule_based_arima_SP_0_05_15_50_short.csv', 
    index_col='date',  # Set the 'date' column as the index
    parse_dates=True   # Tell pandas to convert the index to datetime objects
) 







print("30 year- strategy 1 vs 2")

df_30_1_stats = performance_stats(df_30_1, "2024-09-04", "2024-11-04")
print(df_30_1_stats)
print()
print()

df_30_2_stats = performance_stats(df_30_2, "2024-09-04", "2024-11-04")
print(df_30_2_stats)
print()
print()

print("40 year- strategy 1 vs 2")

df_40_1_stats = performance_stats(df_40_1, "2024-09-04", "2024-11-04")
print(df_40_1_stats)
print()
print()

df_40_2_stats = performance_stats(df_40_2, "2024-09-04", "2024-11-04")
print(df_40_2_stats)
print()
print()

print("50 year- strategy 1 vs 2")

df_50_1_stats = performance_stats(df_50_1, "2024-09-04", "2024-11-04")
print(df_50_1_stats)
print()
print()

df_50_2_stats = performance_stats(df_50_2, "2024-09-04", "2024-11-04")
print(df_50_2_stats)
print()
print()












#rule based approaches
print("30 year- strategy rule based arima")

df_30_arima_stats = performance_stats(df_30_arima, "2024-09-04", "2024-11-04")
print(df_30_arima_stats)
print()
print()



print("40 year- strategy rule based arima")

df_40_arima_stats = performance_stats(df_40_arima, "2024-09-04", "2024-11-04")
print(df_40_arima_stats)
print()
print()

print("50 year- strategy rule based arima")

df_50_arima_stats = performance_stats(df_50_arima, "2024-09-04", "2024-11-04")
print(df_50_arima_stats)
print()
print()







#ablation and longer term


#5 year- strategy 
print("30 year- strategy 1 long dated") #long dated
df_30_1_stats = performance_stats(df_30_1, "2024-06-04", "2024-11-04")
print(df_30_1_stats)
print()
print()

print("30 year- strategy 2 long dated")
df_30_2_stats = performance_stats(df_30_2, "2024-06-04", "2024-11-04")
print(df_30_2_stats)
print()
print()

print("30 year- strategy 2 long dated- no text")
df_30_2_stats_no_text = performance_stats(df_30_2_no_text, "2024-06-04", "2024-11-04")
print(df_30_2_stats_no_text)
print()
print()

print("30 year- strategy 2 long dated- no numeric")
df_30_2_stats_no_numeric = performance_stats(df_30_2_no_numeric, "2024-06-04", "2024-11-04")
print(df_30_2_stats_no_numeric)
print()
print()

print("30 year- strategy 2 long dated- no arima")
df_30_2_stats_no_arima = performance_stats(df_30_2_no_arima, "2024-06-04", "2024-11-04")
print(df_30_2_stats_no_arima)
print()
print()

#rule based approaches
print("30 year- strategy rule based arima")

df_30_arima_stats = performance_stats(df_30_arima, "2024-06-04", "2024-11-04")
print(df_30_arima_stats)
print()
print()


#statistical significance with temp=0.3

print("50 year- strategy rule based arima - trial 1")

df_50_2_trial_1_stats = performance_stats(df_50_2_trial_1, "2024-09-04", "2024-11-04")
print(df_50_2_trial_1_stats)
print()
print()

print("50 year- strategy rule based arima - trial 2")

df_50_2_trial_2_stats = performance_stats(df_50_2_trial_2, "2024-09-04", "2024-11-04")
print(df_50_2_trial_2_stats)
print()
print()

print("50 year- strategy rule based arima - trial 3")

df_50_2_trial_3_stats = performance_stats(df_50_2_trial_3, "2024-09-04", "2024-11-04")
print(df_50_2_trial_3_stats)
print()
print()

print("50 year- strategy rule based arima - trial 4")

df_50_2_trial_4_stats = performance_stats(df_50_2_trial_4, "2024-09-04", "2024-11-04")
print(df_50_2_trial_4_stats)
print()
print()


print("50 year- strategy rule based arima - trial 5")

df_50_2_trial_5_stats = performance_stats(df_50_2_trial_5, "2024-09-04", "2024-11-04")
print(df_50_2_trial_5_stats)
print()
print()

#compute stats

stats_list = [df_50_2_trial_1_stats, df_50_2_trial_2_stats, df_50_2_trial_3_stats, df_50_2_trial_4_stats, df_50_2_trial_5_stats]
aggregated_results = aggregate_performance_stats(stats_list)

print(aggregated_results)