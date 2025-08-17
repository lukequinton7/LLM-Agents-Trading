#Using ARIMA(1,1,1) as a benchmark for inflation predictions
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Configuration
nowcast_csv_file = "Nowcast historic.csv" #historic inflation data
PREDICTION_HORIZON_DAYS = 28 #prediction days forward
START_DATE = "2024-06-03"  # Match LLM data (first Monday)
END_DATE = "2025-05-01"    # Match LLM data (last Monday)

# Load data- to df
nowcast_df = pd.read_csv(nowcast_csv_file)
nowcast_df['Date'] = pd.to_datetime(nowcast_df['Date'], format='%m/%d/%Y')
nowcast_df = nowcast_df.sort_values('Date', ascending=False).reset_index(drop=True)

# Get Mondays only (weekly format- to match llm inflation predicts)
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)

available_mondays = nowcast_df[
    (nowcast_df['Date'] >= start_date) & 
    (nowcast_df['Date'] <= end_date) &
    (nowcast_df['Date'].dt.dayofweek == 0)  # Monday=0
]['Date'].sort_values()

print(f"Processing {len(available_mondays)} Mondays")

# Main Processing 
all_predictions = []

for current_date in available_mondays:
    current_date = current_date.date()
    print(f"Processing: {current_date}")
    
    # Target date
    current_date_dt = pd.to_datetime(current_date)
    target_prediction_date = current_date_dt + timedelta(days=PREDICTION_HORIZON_DAYS)
    
    # Get 100 days of historical data
    end_date_window = current_date_dt - timedelta(days=1)
    start_date_window = end_date_window - timedelta(days=100)
    
    historical_data = nowcast_df[
        (nowcast_df['Date'] >= start_date_window) & 
        (nowcast_df['Date'] <= end_date_window)
    ].sort_values('Date')
    
    # Fit ARIMA(1,1,1) and predict
    series = historical_data['Nowcast'].values
    model = ARIMA(series, order=(1,1,1))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=PREDICTION_HORIZON_DAYS)
    arima_prediction = round(forecast[-1], 2)
    
    # Get actual value
    actual_rows = nowcast_df[nowcast_df['Date'] == target_prediction_date.date()]
    actual_nowcast = actual_rows.iloc[0]['Nowcast'] if not actual_rows.empty else None
    
    # Store with DD/MM/YYYY format to match LLM data
    all_predictions.append({
        "prediction_date": current_date.strftime('%d/%m/%Y'),
        "target_date": target_prediction_date.strftime('%d/%m/%Y'),
        "prediction": arima_prediction,
        "initial_prediction": arima_prediction,  # Same as prediction for ARIMA
        "actual_nowcast": actual_nowcast,
        "num_summaries_used": 0  # ARIMA doesn't use summaries
    })
    
    print(f"  ARIMA Prediction: {arima_prediction}")
    print()
    print()

# Save results with matching filename format
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv("arima_predictions_inflation_28day_benchmark.csv", index=False)
print(f"\nSaved {len(all_predictions)} predictions")

# Calculate MAE- rough eye test
df_complete = predictions_df.dropna(subset=['actual_nowcast'])
if len(df_complete) > 0:
    mae = abs(df_complete['prediction'] - df_complete['actual_nowcast']).mean()
    print(f"ARIMA MAE: {mae:.4f}")
else:
    print("ARIMA MAE: No complete data for evaluation")