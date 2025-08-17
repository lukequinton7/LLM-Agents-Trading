import pandas as pd
import numpy as np
import clean_input_df as input_

START_DATE = "2024-09-04"  #  start date
END_DATE = "2024-11-04"    # end date
treasury_name = "SP 0 05 15 50" #25yr

STRIPS = input_.STRIPS_to_df(0)
pred = input_.ARIMA_to_df(treasury_name)

def cap_df(df, date_start, date_end):
    """
    function to custom range a df with df input 
    """
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    
    # filter dataframe between dates (inclusive)
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # aort by date (ascending - oldest first)
    filtered_df = filtered_df.sort_index()
    
    return filtered_df

STRIPS = cap_df(STRIPS, START_DATE, END_DATE)
pred = cap_df(pred, START_DATE, END_DATE)

print(STRIPS.tail(20))
print(pred.tail(20))

results = ["date", "recommendation", "position_size", "price", "new_position"]
results_df = pd.DataFrame(columns=results)

# Trading parameters
T = 0.000  # 0.0% threshold
current_position = 0

# Loop through df by date
for date in STRIPS.index:
    current_price = STRIPS.loc[date, 'Last Px']
    pred_1 = pred.loc[date, 'pred_1']
    pred_7 = pred.loc[date, 'pred_7']
    pred_28 = pred.loc[date, 'pred_28']
    
    # If predictions T% higher than current price BUY
    if (pred_1 > current_price * (1 + T) and 
        pred_7 > current_price * (1 + T)):
        
        if current_position > 0:
            recommendation = "HOLD"
            position_size = 0
        else:
            recommendation = "BUY"
            current_position += 1
            position_size = 1

    elif (pred_1 < current_price * (1 - T) and 
          pred_7 < current_price * (1 - T)):
        
        if current_position < 0:
            recommendation = "HOLD"
            position_size = 0
        else:
            recommendation = "SELL"
            current_position -= 1
            position_size = -1
    else:
        recommendation = "HOLD"
        position_size = 0
    
    # Add to results
    new_row = pd.DataFrame({
        "date": [date],
        "recommendation": [recommendation],
        "position_size": [position_size],
        "price": [current_price],
        "new_position": [current_position]
    })
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)

pd.set_option('display.max_rows', None)
print(results_df)

results_df.to_csv(f"rule_based_arima_{treasury_name.replace(' ', '_')}_short.csv", index=False)