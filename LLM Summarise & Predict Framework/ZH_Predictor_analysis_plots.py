import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "weekly_predictions_inflation_gemini-2_5-pro_with_reflection_14 day.csv"

df_prediction = pd.read_csv(csv_file)
df_prediction['prediction_date'] = pd.to_datetime(df_prediction['prediction_date'], format='%d/%m/%Y')
df_prediction['target_date'] = pd.to_datetime(df_prediction['target_date'], format='%d/%m/%Y')

# Calculate errors
df_prediction['reflection_error'] = abs(df_prediction['prediction'] - df_prediction['actual_nowcast'])
df_prediction['initial_error'] = abs(df_prediction['initial_prediction'] - df_prediction['actual_nowcast'])
df_prediction['reflection_active'] = df_prediction['prediction'] != df_prediction['initial_prediction']

# Remove rows with missing actual data for error plots
df_complete = df_prediction.dropna(subset=['actual_nowcast'])

def plot_predictions(df):
    """
    Create 2 plots: time series (1.75:1 ratio) and error comparison (square fig)
    """
    
    fig = plt.figure(figsize=(16, 6))
    
    # plot 1: Time Series (1.75:1 width:height ratio)
    ax1 = plt.subplot(1, 9, (1, 5))  # Takes first 5 columns for 1.75:1 ratio
    ax1.plot(df['prediction_date'], df['prediction'], '--', color='black', 
             label='Reflection Agent Prediction (28 day lag)', linewidth=2)
    ax1.plot(df['prediction_date'], df['actual_nowcast'], '-', color='green', 
             label='Actual- Cleveland Nowcast CPI (YoY)', linewidth=2)
    
    ax1.set_title('28 Day Nowcast Inflation Predictions')
    ax1.set_xlabel('Prediction Date')
    ax1.set_ylabel('Inflation Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # plot 2: Error Comparison (approximately square)
    ax2 = plt.subplot(1, 9, (7, 9))  # Takes columns 7-9 for square ratio
    width = 0.35
    x = np.arange(len(df_complete))
    
    ax2.bar(x - width/2, df_complete['reflection_error'], width, 
           label='Reflection Agent Error', color='green', alpha=0.7)
    ax2.bar(x + width/2, df_complete['initial_error'], width,
           label='Prediction Agent Error', color='black', alpha=0.7)
    
    # horizontal lines for MAE
    reflection_mae = df_complete['reflection_error'].mean()
    initial_mae = df_complete['initial_error'].mean()
    
    ax2.axhline(y=reflection_mae, color='red', linestyle='--', linewidth=1.2, alpha=0.8, 
               label=f'Reflection Agent MAE ({reflection_mae:.3f})')
    ax2.axhline(y=initial_mae, color='black', linestyle='--', linewidth=1.2, alpha=0.8,
               label=f'Prediction Agent MAE ({initial_mae:.3f})')
    
    ax2.set_title('Prediction Errors')
    ax2.set_xlabel('Prediction Index')
    ax2.set_ylabel('Absolute Error')
    ax2.legend(fontsize = 8)
    ax2.grid(True, alpha=0.3)

    #set y axis height to 0.5 exactly
    ax2.set_ylim(0, 0.72)    

    
    plt.tight_layout()
    return fig, (ax1, ax2)

def compute_statistics(df):
    """
    Compute summary statistics and return as dictionary
    """
    df_complete = df.dropna(subset=['actual_nowcast'])
    
    reflection_mae = df_complete['reflection_error'].mean()
    initial_mae = df_complete['initial_error'].mean()
    reflection_rmse = np.sqrt((df_complete['reflection_error']**2).mean())
    initial_rmse = np.sqrt((df_complete['initial_error']**2).mean())
    
    # Calculate regret (how often reflection agent performs worse)
    regret_count = (df_complete['reflection_error'] > df_complete['initial_error']).sum()
    regret_percentage = (regret_count / len(df_complete)) * 100
    
    total_predictions = len(df)
    reflection_active = df['reflection_active'].sum()
    reflection_active_pct = (reflection_active / total_predictions) * 100
    
    improvement_mae = ((initial_mae - reflection_mae) / initial_mae * 100) if initial_mae > 0 else 0
    improvement_rmse = ((initial_rmse - reflection_rmse) / initial_rmse * 100) if initial_rmse > 0 else 0
    
    stats = {
        'total_predictions': total_predictions,
        'reflection_active_count': reflection_active,
        'reflection_active_percentage': reflection_active_pct,
        'reflection_mae': reflection_mae,
        'initial_mae': initial_mae,
        'improvement_mae_percent': improvement_mae,
        'reflection_rmse': reflection_rmse,
        'initial_rmse': initial_rmse,
        'improvement_rmse_percent': improvement_rmse,
        'regret_count': regret_count,
        'regret_percentage': regret_percentage,
        'start_date': df['prediction_date'].min().strftime('%d/%m/%Y'),
        'end_date': df['prediction_date'].max().strftime('%d/%m/%Y'),
        'inflation_min': df['actual_nowcast'].min(),
        'inflation_max': df['actual_nowcast'].max()
    }
    
    return stats

# Create plots
fig, axes = plot_predictions(df_prediction)
plt.show()

# Compute statistics
stats = compute_statistics(df_prediction)
print("Summary Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")