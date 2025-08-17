import pandas as pd
from datetime import timedelta, datetime
import clean_input_df as input_
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Configuration
USE_120MIN_DATA = False  # Set to True for 120-minute data, False for daily
PREDICTION_HORIZON_DAYS = 28  # 1 or 7 days ahead
treasury_name = "SP 0 05 15 50"

if USE_120MIN_DATA:
    HISTORIC_DATA_PERIODS = 48  # 12 periods = 24 hours of intraday data
    strips_df = input_.STRIPS_120mins_to_df()
    column_name = 'Last PX'
else:
    HISTORIC_DATA_PERIODS = 100   
    strips_df = input_.STRIPS_to_df()
    column_name = 'Last Px'

START_DATE = "2024-09-04"
END_DATE = "2024-11-04"
ARIMA_ORDER = (1, 1, 1)

def fit_arima_and_predict(price_series, horizon_days, order):
    """Fit ARIMA and predict"""
    try:
        model = ARIMA(price_series, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=horizon_days)
        return round(forecast.iloc[-1], 3)
    except Exception as e:
        print(f"ARIMA fitting error: {e}")
        return round(price_series.mean(), 2)

def get_bond_price_series(current_date_str, historic_periods):
    """Get historical bond prices with data availability rules"""
    try:
        current_date = pd.to_datetime(current_date_str)
        
        if USE_120MIN_DATA:
            # 120-min: same day up to 6 PM + all previous days
            start_date = current_date - timedelta(hours=historic_periods * 2)
            historical_data = strips_df[strips_df.index >= start_date].sort_index()
            
            # Apply 10 PM cutoff for current day
            current_day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            current_day_6pm = current_date.replace(hour=22, minute=0, second=0, microsecond=0)
            
            mask = (historical_data.index < current_day_start) | \
                   ((historical_data.index >= current_day_start) & (historical_data.index <= current_day_6pm))
            historical_data = historical_data[mask]
        else:
            # Daily: current day + previous days
            start_date = current_date - timedelta(days=historic_periods)
            historical_data = strips_df[
                (strips_df.index >= start_date) & (strips_df.index <= current_date)
            ].sort_index()
        
        return historical_data[column_name] if not historical_data.empty else pd.Series([80.0] * 10)
        
    except Exception as e:
        print(f"Data extraction error: {e}")
        return pd.Series([80.0] * 10)

def calculate_target_date(current_date, horizon_days):
    """Calculate target date - handle weekends for both data types"""
    if current_date.weekday() == 4 and horizon_days == 1:  # Friday predicting 1 day ahead
        return current_date + timedelta(days=3)  # Friday -> Monday
    else:
        return current_date + timedelta(days=horizon_days)

# Main Processing Loop 
all_predictions = []

# Generate date range
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)
current_date = start_date

print(f"Starting ARIMA bond predictions from {START_DATE} to {END_DATE}")
print(f"Treasury: {treasury_name}")
print(f"Prediction horizon: {PREDICTION_HORIZON_DAYS} days")
print(f"ARIMA order: {ARIMA_ORDER}")
print()
print()

while current_date <= end_date:
    # Always use daily date format for iteration, regardless of data frequency
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    target_date = calculate_target_date(current_date, PREDICTION_HORIZON_DAYS)
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"Processing: {current_date_str}   Target: {target_date_str}")
    
    try:
        # Get historical bond prices
        price_series = get_bond_price_series(current_date_str, HISTORIC_DATA_PERIODS)
        
        # Fit ARIMA and make prediction
        arima_prediction = fit_arima_and_predict(
            price_series, 
            horizon_days=PREDICTION_HORIZON_DAYS, 
            order=ARIMA_ORDER
        )
        
        print(f" ARIMA Prediction: {arima_prediction}")
        
        # Always get actual price from daily STRIPS data (regardless of input data type)
        daily_strips_df = input_.STRIPS_to_df()  # Always use daily data for actual price
        actual_price = input_.get_actual_bond_price(daily_strips_df, target_date_str)
        
        # Store results 
        prediction_result = {
            'date': current_date_str,
            'target_date': target_date_str,
            'prediction': arima_prediction,
            'reflection': arima_prediction,
            'actual_price': actual_price
        }
        
        all_predictions.append(prediction_result)
        
        print(f"   Results saved for {current_date_str}")
        
    except Exception as e:
        print(f"   ERROR: ARIMA prediction failed for {current_date_str}: {e}")
        print(f"  Skipping this date and continuing...")
        print()

    # Move to next day (same logic for both data types)
    current_date += timedelta(days=3 if current_date.weekday() == 4 else 1) 

# save results to CSV
if all_predictions:
    results_df = pd.DataFrame(all_predictions)
    frequency_suffix = "120min" if USE_120MIN_DATA else "daily"
    output_filename = f"bond_predictions_ARIMA_{treasury_name.replace(' ', '_')}_{frequency_suffix}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nSaved {len(all_predictions)} ARIMA predictions to {output_filename}")
    print(f"Columns: {list(results_df.columns)}")
    
    # performance summary prints
    valid_predictions = results_df.dropna(subset=['actual_price'])
    if len(valid_predictions) > 0:
        mae = abs(valid_predictions['prediction'] - valid_predictions['actual_price']).mean()
        print(f"Mean Absolute Error: {mae:.4f}")
else:
    print("No predictions generated.")