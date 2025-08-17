import pandas as pd
from typing import TypedDict, Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from datetime import timedelta
import os

# Configuration 
GEMINI_API_KEY = "" #left blank for privacy
summaries_csv_file = "ZH_article_summaries_full_inflation_gemma3_12b.csv" #agent summarised articles (gemma 3-12b)
nowcast_csv_file = "Nowcast historic.csv" #inflation data- historic
model_select = "gemini-2.5-pro"
model_name = str(model_select).replace(":", "_").replace("/", "_").replace(".", "_") # model string change for save
max_token_limit = 16000
TASK_TYPE = "inflation" #save
PREDICTION_HORIZON_DAYS = 14 #days forward to predict
MAX_PREDICTION_HISTORY = 21  # Number of previous predictions to keep for reflection
NEWS_LOOKBACK_DAYS = 3  # Number of days to look back for news summaries
ABLATION_NO_SUMMARIES = True # keep news summaries or not

# Manual date range configuration
START_DATE = "2024-06-01"  # Manual start date
END_DATE = "2025-05-01"    # Manual end date

# Data Loading - to df
def load_summaries():
    summaries_df = pd.read_csv(summaries_csv_file)
    summaries_df['datetime'] = pd.to_datetime(summaries_df['datetime'])
    return summaries_df

def load_nowcast_data():
    nowcast_df = pd.read_csv(nowcast_csv_file)
    # Handle MM/DD/YYYY format explicitly
    nowcast_df['Date'] = pd.to_datetime(nowcast_df['Date'], format='%m/%d/%Y')
    nowcast_df = nowcast_df.sort_values('Date', ascending=False).reset_index(drop=True)
    return nowcast_df

def get_historical_nowcast(nowcast_df, reference_date, days_back=100):
    # ensure date format
    reference_date = pd.to_datetime(reference_date)
    # calc lookback start date
    start_date = reference_date - timedelta(days=days_back)
    
    # filter df for our date range
    historical_data = nowcast_df[
        (nowcast_df['Date'] >= start_date) & 
        (nowcast_df['Date'] <= reference_date)
    ].sort_values('Date')
    
    # handle empty df
    if historical_data.empty:
        return "No nowcast data available"
    
    # build output string - header
    nowcast_text = f"Historical Nowcast Inflation Data (last {len(historical_data)} readings):\n\n"
    # loop and format each row
    for _, row in historical_data.iterrows():
        nowcast_text += f"{row['Date'].strftime('%Y-%m-%d')}: {row['Nowcast']}%\n"
    
    # add trend if enough data
    if len(historical_data) >= 2:
        latest_value = historical_data.iloc[-1]['Nowcast']
        oldest_value = historical_data.iloc[0]['Nowcast']
        change = latest_value - oldest_value
        nowcast_text += f"\nTrend: {change:+.2f}% (from {oldest_value}% to {latest_value}%)"
    
    return nowcast_text

def get_actual_nowcast_for_date(nowcast_df, target_date, current_simulation_date=None):
    """Get actual nowcast value for a specific date, only if it's available by the current simulation date"""
    target_date = pd.to_datetime(target_date)
    
    if current_simulation_date is not None:
        current_simulation_date = pd.to_datetime(current_simulation_date)
        if target_date > current_simulation_date:
            return None
    
    matching_rows = nowcast_df[nowcast_df['Date'] == target_date]
    
    if not matching_rows.empty:
        return matching_rows.iloc[0]['Nowcast']
    return None

def format_previous_predictions(prediction_history, nowcast_df, current_simulation_date=None):
    """Format previous predictions with actual values for comparison"""
    if not prediction_history:
        return "No previous predictions available."
    
    formatted_text = "Previous Agent Predictions vs Actual Results:\n\n"
    
    for i, pred in enumerate(prediction_history):
        prediction_date = pred['prediction_date']
        target_date = pred['target_date']
        initial_prediction = pred.get('initial_prediction', pred['prediction'])  # fallback for old format
        final_prediction = pred['prediction']
        
        actual_value = get_actual_nowcast_for_date(nowcast_df, target_date, current_simulation_date)
        
        formatted_text += f"Prediction Date: {prediction_date}\n"
        formatted_text += f"Target Date: {target_date}\n"
        formatted_text += f"Predictor agent: {initial_prediction}%\n"
        
        # For the current prediction being made, don't show reflection value yet
        if i == len(prediction_history) - 1:
            formatted_text += f"Predictor agent (after reflection): [current prediction being made]\n"
        else:
            formatted_text += f"Predictor agent (after reflection): {final_prediction}%\n"
        
        if actual_value is not None:
            formatted_text += f"Actual: {actual_value}%\n"
        else:
            formatted_text += "Actual: Not yet available\n"
        
        formatted_text += "---\n"
    
    return formatted_text

#impose hard cap in case gemma 3 summaries ignored previous cap in prompt
def truncate_individual_summaries(summaries_list, max_words_per_summary=200):
    """Truncate each individual summary to max_words_per_summary"""
    truncated_summaries = []
    
    for summary in summaries_list:
        words = summary.split()
        if len(words) <= max_words_per_summary:
            truncated_summaries.append(summary)
        else:
            truncated_summary = ' '.join(words[:max_words_per_summary]) + "..."
            truncated_summaries.append(truncated_summary)
    
    return truncated_summaries

summaries_df = load_summaries()
nowcast_df = load_nowcast_data()

# LangGraph State Definition
class PredictorState(TypedDict):
    summary_context: str
    nowcast_data: str
    prediction_date: str
    target_prediction_date: str
    prediction_history: List[Dict]
    initial_prediction: Optional[str]
    final_prediction: Optional[str]

# Model Setup- works with Gemini variants (eg. 1.5, 2.5 pro etc)
model = ChatGoogleGenerativeAI(
    model=model_select,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_output_tokens=max_token_limit
)

#prediction template
inflation_predictor_template = '''
Today is {current_date}.

You are an inflation prediction agent designed to receive relevant inputs and predict inflation in {horizon} days time (particularly Nowcast, Bloomberg ticker: CLEVINF Index).

You will use present and historical nowcast inflation data and also inflation news article summaries to make your prediction.

You must output a single numerical value for your prediction, which is the expected nowcast inflation rate in {horizon} days time.

IMPORTANT: your output must be a numerical value with no additional text, and it must be formatted to 2 decimal places.

Here are some correctly and incorrectly formatted responses for your reference:

example 1 (correct-2dp):
3.70

example 2 (correct-2dp):
0.79

example 3 (incorrect-1dp):
4.1

example 4 (incorrect- percentage sign not allowed):
0.11% 

Here is the previous nowcast inflation data.....
{nowcast_data}

Here are the relevant news article summaries from the last {news_days} days.....
{summary_context}
'''

#reflection template
inflation_reflection_template = '''
Today is {current_date}.

You are an inflation prediction agent designed to reflect on another agent's predictions and improve the output.

The previous agent has predicted nowcast inflation for {horizon} days in the future.

You must compare the previous agent's predictions with the actual nowcast inflation data (matching prediction for date with realised nowcast date).

eg. prediction date: 2023-01-01, prediction for 2023-15-01: 3.70, compared to actual nowcast inflation for 2023-15-01: 3.50...

You must output a single numerical value for your prediction, which is the expected nowcast inflation rate in {horizon} days time.

IMPORTANT: your output must be a numerical value with no additional text, and it must be formatted to 2 decimal places.

Here are some correctly and incorrectly formatted responses for your reference:

example 1 (correct-2dp):
3.70

example 2 (correct-2dp):
0.79

example 3 (incorrect-1dp):
4.1

example 4 (incorrect- percentage sign not allowed):
0.11% 

Here is the previous nowcast inflation data.....
{nowcast_data}

Here are the previous agent's predictions:
{previous_predictions}

If there is not yet any nowcast inflation data available, you must output the previous agent's prediction (unchanged).
'''

# Agent Nodes
def predictor_node(state: PredictorState) -> PredictorState:
    prompt = inflation_predictor_template.format(
        current_date=state['prediction_date'],
        horizon=PREDICTION_HORIZON_DAYS,
        news_days=NEWS_LOOKBACK_DAYS,
        summary_context=state['summary_context'], 
        nowcast_data=state['nowcast_data']
    )
    
    response = model.invoke(prompt)
    state['initial_prediction'] = str(response.content).strip()
    print(f"    Predictor Agent Output: {state['initial_prediction']}")
    return state

def reflection_node(state: PredictorState) -> PredictorState:
    current_date = pd.to_datetime(state['prediction_date'])
    previous_predictions_text = format_previous_predictions(
        state['prediction_history'], 
        nowcast_df, 
        current_simulation_date=current_date
    )
    
    if not state['prediction_history'] or "No previous predictions" in previous_predictions_text:
        state['final_prediction'] = state['initial_prediction']
        print(f"    → Reflection Agent Output: {state['final_prediction']} (no history, using initial prediction)")
    else:
        # Check if we have realised values
        realised_checks = []
        for pred in state['prediction_history']:
            actual_value = get_actual_nowcast_for_date(nowcast_df, pred['target_date'], current_date)
            realised_checks.append({
                'target_date': pred['target_date'],
                'actual_value': actual_value,
                'has_value': actual_value is not None
            })
            print(f"      Debug: Target date {pred['target_date']} -> Actual value: {actual_value}")
        
        has_realised_values = any(check['has_value'] for check in realised_checks)
        
        if has_realised_values:
            print(f"    Reflection Agent: Found realised values, analysing predictions...")
            prompt = inflation_reflection_template.format(
                current_date=state['prediction_date'],
                horizon=PREDICTION_HORIZON_DAYS,
                nowcast_data=state['nowcast_data'],
                previous_predictions=previous_predictions_text
            )
            
            response = model.invoke(prompt)
            state['final_prediction'] = str(response.content).strip()
            print(f"    Reflection Agent Output: {state['final_prediction']} (adjusted based on history)")
        else:
            state['final_prediction'] = state['initial_prediction']
            print(f"    Reflection Agent Output: {state['final_prediction']} (history exists but no realised values yet as of {current_date.date()})")
    
    return state

# Build graph
predictor_graph_builder = StateGraph(PredictorState)
predictor_graph_builder.add_node("predictor", predictor_node)
predictor_graph_builder.add_node("reflector", reflection_node)

# Define the flow: START -> predictor -> reflector -> END
predictor_graph_builder.add_edge(START, "predictor")
predictor_graph_builder.add_edge("predictor", "reflector")
predictor_graph_builder.add_edge("reflector", END)

predictor_agent = predictor_graph_builder.compile()

# Main Processing Logic
all_predictions = []
prediction_history = []

# Get Mondays between start and end date that exist in nowcast data- ie for weekly runs
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)

# Get all nowcast dates that fall within our range and are Mondays
available_mondays = nowcast_df[
    (nowcast_df['Date'] >= start_date) & 
    (nowcast_df['Date'] <= end_date) &
    (nowcast_df['Date'].dt.dayofweek == 0)  # Monday=0
]['Date'].sort_values()

print(f"Processing {len(available_mondays)} Mondays from {START_DATE} to {END_DATE}")
print(f"Available Mondays: {[d.date() for d in available_mondays]}")

#loop through dats/weeks and invoke agents
for current_date in available_mondays:
    current_date = current_date.date()  # Convert to date for consistency
    print(f"Processing: {current_date}")
    
    # Calculate target prediction date
    current_date_dt = pd.to_datetime(current_date)
    target_prediction_date = current_date_dt + timedelta(days=PREDICTION_HORIZON_DAYS)
    
    # Get news summary window
    start_date_window = current_date_dt - timedelta(days=NEWS_LOOKBACK_DAYS)
    end_date_window = current_date_dt - timedelta(days=1)
    
    # Filter relevant summaries
    all_summaries_in_window = summaries_df[
        (summaries_df['datetime'].dt.date >= start_date_window.date()) &
        (summaries_df['datetime'].dt.date <= end_date_window.date())
    ]
    
    relevant_summaries_df = all_summaries_in_window[
        ~all_summaries_in_window['summary'].str.contains('no relevant content', case=False, na=False)
    ]
    
    print(f"  Date window: {start_date_window.date()} to {end_date_window.date()}")
    print(f"  Total articles in window: {len(all_summaries_in_window)}")
    print(f"  Relevant articles (after filtering): {len(relevant_summaries_df)}")
    print(f"  Target prediction date: {target_prediction_date.date()}")
    
    if len(relevant_summaries_df) == 0:
        print("  No relevant summaries found, skipping...")
        continue
    
    # Truncate each individual summary to 200 words but keep all relevant ones
    summaries_list = relevant_summaries_df['summary'].astype(str).tolist()
    truncated_summaries = truncate_individual_summaries(summaries_list, max_words_per_summary=200)
    summary_context = "\n\n---\n\n".join(truncated_summaries)



    # ABLATION TEST
    if ABLATION_NO_SUMMARIES:
        summary_context = "NO ARTICLES" #replace prompt text with "no articles"
    else:
        print(f"  All {len(truncated_summaries)} summaries included (each capped at 200 words)")


    
    print(f"  All {len(truncated_summaries)} relevant summaries included (each capped at 200 words)")
    
    # Get nowcast data (up to end of window, not current date)
    nowcast_data = get_historical_nowcast(nowcast_df, end_date_window, days_back=100)
    
    # Prepare state with prediction history
    initial_state = {
        "summary_context": summary_context,
        "nowcast_data": nowcast_data,
        "prediction_date": str(current_date),
        "target_prediction_date": str(target_prediction_date.date()),
        "prediction_history": prediction_history.copy(),
        "initial_prediction": None,
        "final_prediction": None
    }
    
    try:
        # Run the agent (predictor -> reflector)
        final_state = predictor_agent.invoke(initial_state)
        
        # Store the prediction in our history for future reflection
        # Also get the actual nowcast value for this target date (if available)
        actual_nowcast = get_actual_nowcast_for_date(nowcast_df, target_prediction_date.date(), None)  # No date restriction for final CSV
        
        prediction_entry = {
            "prediction_date": str(current_date),
            "target_date": str(target_prediction_date.date()),
            "prediction": final_state['final_prediction'],
            "initial_prediction": final_state['initial_prediction'],
            "actual_nowcast": actual_nowcast,
            "num_summaries_used": len(truncated_summaries)
        }
        
        # Add to prediction history for future reflections
        prediction_history.append({
            "prediction_date": str(current_date),
            "target_date": str(target_prediction_date.date()),
            "prediction": final_state['final_prediction'],
            "initial_prediction": final_state['initial_prediction']
        })
        
        # Keep only recent predictions for reflection
        if len(prediction_history) > MAX_PREDICTION_HISTORY:
            prediction_history = prediction_history[-MAX_PREDICTION_HISTORY:]
        
        all_predictions.append(prediction_entry)
        print(f" Initial Prediction: {final_state['initial_prediction']}")
        print(f" Final Prediction (after reflection): {final_state['final_prediction']}")
        print(f" Prediction stored for {target_prediction_date.date()}")
        print()
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        continue

# Save results
if all_predictions:
    predictions_df = pd.DataFrame(all_predictions)
    output_filename = f"weekly_predictions_{TASK_TYPE}_{model_name}_with_reflection_ABLATION.csv" #ABLATION IF NECCESARY
    predictions_df.to_csv(output_filename, index=False)
    print(f"\nSaved {len(all_predictions)} predictions to {output_filename}")
    print(f"Columns: {list(predictions_df.columns)}")
else:
    print("No predictions generated.")