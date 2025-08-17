import pandas as pd
from typing import TypedDict, Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from datetime import timedelta, datetime
import clean_input_df as input_ #df cleaning and utility functions
import time


# Configuration 
GEMINI_API_KEY = "" # blank for privacy
summaries_csv_file = "ZH_article_summaries_full_inflation_gemma3_12b.csv"
nowcast_csv_file = "Nowcast historic.csv"
model_select = "gemini-2.5-pro"
model_name = str(model_select).replace(":", "_").replace("/", "_").replace(".", "_")
max_token_limit = 20000
PREDICTION_HORIZON_DAYS = 14
MAX_PREDICTION_HISTORY = 21  # Number of previous predictions to keep for reflection
HISTORIC_DATA_DAYS = 30  # Number of days of historic data to use for rates and inflation
HISTORIC_DATA_DAYS_BOND = 100  # Number of days of historic data to use for rates and inflation
HISTORIC_DATA_DAYS_STATEMENT = 50
#treasury_name = "SP 0 05 15 50" #25yr
#treasury_name = "SP 0 05 15 40" #15yr
treasury_name = "SP 0 05 15 30" #5yr

ABLATION_NO_SUMMARIES = True

# Manual date range configuration
START_DATE = "2024-06-04"  # Manual start date
END_DATE = "2025-05-14"    # Manual end date

# Additional configs
MAX_RETRIES = 3  # API retry attempts
INITIAL_RETRY_DELAY = 5  # Initial retry delay in seconds
BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

#load cleaned historic dataframes
strips_df = input_.STRIPS_to_df() #historic STRIPS data
rates_df = input_.EFFR_to_df()  #historic rate data
nowcast_df = input_.Nowcast_to_df() #historic nowcast data
FOMC_df = input_.FOMC_PDFs_to_df() #historic FOMC statements

# --- Enhanced LangGraph State Definition ---
class PredictorState(TypedDict):
    current_date: str
    target_date: str
    strips_data: str
    rates_data: str
    nowcast_data: str
    fomc_data: str
    prediction_agent_output: Optional[str]
    reflection_agent_output: Optional[str]
    predictions_history: List[Dict]

# --- Model Setup ---
model = ChatGoogleGenerativeAI(
    model=model_select,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_output_tokens=max_token_limit
)

bond_predictor_template = '''
Today is {current_date}.

You are a bond prediction agent designed to receive relevant inputs and predict a zero coupon US Treasury price in {horizon} days time.

The particular Treasury to predict is {treasury_name}.

You will use present and historical nowcast inflation data, Fed interest rate data, Treasury prices (IMPORTANT same as you are predicting),
and also FOMC Statements to make your prediction.

You must output a single numerical value for your prediction, which is the expected Treasury price rate in {horizon} days time.

IMPORTANT: your output must be a numerical value with no additional text, and it must be formatted to 2 decimal places.

Here are some correctly and incorrectly formatted responses for your reference:

example 1 (correct-2dp):
81.53

example 2 (correct-2dp):
79.91

example 3 (incorrect-1dp):
85.1

example 4 (incorrect- additional text not allowed):
67.87 mid price

Here is the previous Treasury price and yield data (IMPORTANT).....
{strips_data}

Here is the previous FED rates data.....
{rates_data}

Here is the previous nowcast inflation data.....
{nowcast_data}

Here is the most recent FOMC statements.....
{fomc_data}
'''

bond_reflection_template = '''
Today is {current_date}.

You are a bond prediction agent designed to reflect on a previous agents prediction on a zero coupon US Treasury price in {horizon} days time.

The particular Treasury to predict is {treasury_name}.

You will compare the previous agent's predictions to the actual prices and try to identify patterns in the errors and IMPROVE the current prediction if neccessary.

You will have access to the same data as the previous predictor agent: present and historical nowcast inflation data, Fed interest rate data, 
Treasury prices (IMPORTANT same as you are predicting), and also FOMC Statements to make your prediction.

You must output a single numerical value for your prediction, which is the expected Treasury price rate in {horizon} days time.

IMPORTANT: your output must be a numerical value with no additional text, and it must be formatted to 2 decimal places.

Here are some correctly and incorrectly formatted responses for your reference:

example 1 (correct-2dp):
81.53

example 2 (correct-2dp):
79.91

example 3 (incorrect-1dp):
85.1

example 4 (incorrect- additional text not allowed):
67.87 mid price

Here is the current agent's prediction for {horizon} days time.....
{prediction_agent_current}

Here is the current agent's previous predictions, your previous predictions, and the actual realised bond prices...
{predictions_data}

Here is the previous Treasury price and yield data.....
{strips_data}

Here is the previous FED rates data.....
{rates_data}

Here is the previous nowcast inflation data.....
{nowcast_data}

Here is the most recent FOMC statements.....
{fomc_data}
'''

# --- Agent Nodes ---
def prediction_agent_node(state: PredictorState) -> PredictorState:
    prompt = bond_predictor_template.format(
        current_date=state['current_date'],
        horizon=PREDICTION_HORIZON_DAYS,
        treasury_name=treasury_name,
        strips_data=state['strips_data'],
        rates_data=state['rates_data'],
        nowcast_data=state['nowcast_data'],
        fomc_data=state['fomc_data']
    )
    
    response = model.invoke(prompt)
    state['prediction_agent_output'] = str(response.content).strip()
    print(f" Prediction Agent: {state['prediction_agent_output']}")
    return state

def reflection_agent_node(state: PredictorState) -> PredictorState:
    # Format previous predictions history with time-aware actual prices
    predictions_data = "No previous predictions available."
    if state['predictions_history']:
        current_dt = pd.to_datetime(state['current_date'])
        formatted_predictions = []
        
        for pred in state['predictions_history'][-MAX_PREDICTION_HISTORY:]:  
            target_dt = pd.to_datetime(pred['target_date'])
            
            # Only show actual price if current date has passed the prediction target date
            if current_dt > target_dt:
                actual_price = input_.get_actual_bond_price(strips_df, pred['target_date'])
                actual_str = str(actual_price) if actual_price is not None else "N/A"
            else:
                actual_str = "N/A"  # Future date - reflection agent cannot see
            
            formatted_predictions.append(
                f"Date: {pred['date']}, Prediction Date: {pred['target_date']}, "
                f"Prediction: {pred['prediction']}, Reflection: {pred['reflection']}, "
                f"Actual: {actual_str}"
            )
        
        predictions_data = "\n".join(formatted_predictions)
    
    # TEST PRINT: Show what reflection agent is getting
    print(f"  REFLECTION AGENT SEES: {predictions_data[:400]}...")
    
    prompt = bond_reflection_template.format(
        current_date=state['current_date'],
        horizon=PREDICTION_HORIZON_DAYS,
        treasury_name=treasury_name,
        prediction_agent_current=state['prediction_agent_output'],
        predictions_data=predictions_data,
        strips_data=state['strips_data'],
        rates_data=state['rates_data'],
        nowcast_data=state['nowcast_data'],
        fomc_data=state['fomc_data']
    )
    
    
    response = model.invoke(prompt)
    state['reflection_agent_output'] = str(response.content).strip()
    print(f" Reflection Agent: {state['reflection_agent_output']}")
    return state

# Build graph
graph_builder = StateGraph(PredictorState)
graph_builder.add_node("predictor", prediction_agent_node)
graph_builder.add_node("reflector", reflection_agent_node)

graph_builder.add_edge(START, "predictor")
graph_builder.add_edge("predictor", "reflector")
graph_builder.add_edge("reflector", END)

bond_prediction_agent = graph_builder.compile()



# --- Main Processing Loop ---
all_predictions = []
predictions_history = []

# Generate date range
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)
current_date = start_date

print(f"Starting bond predictions from {START_DATE} to {END_DATE}")
print(f"Treasury: {treasury_name}")
print(f"Prediction horizon: {PREDICTION_HORIZON_DAYS} days")

while current_date <= end_date:
    current_date_str = current_date.strftime('%Y-%m-%d')
    target_date = current_date + timedelta(days=PREDICTION_HORIZON_DAYS)
    target_date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"Processing: {current_date_str} Target: {target_date_str}")
    
    # Get last n days of data from each source
    strips_data = input_.get_last_n_days_data(strips_df, current_date_str, HISTORIC_DATA_DAYS_BOND)
    rates_data = input_.get_last_n_days_data(rates_df, current_date_str,HISTORIC_DATA_DAYS)
    nowcast_data = input_.get_last_n_days_data(nowcast_df, current_date_str,HISTORIC_DATA_DAYS)
    fomc_data = input_.get_last_n_days_data(FOMC_df, current_date_str, HISTORIC_DATA_DAYS_STATEMENT)  # just 1 report normally FOMC
    
    # Prepare state
    initial_state = {
        'current_date': current_date_str,
        'target_date': target_date_str,
        'strips_data': strips_data,
        'rates_data': rates_data,
        'nowcast_data': nowcast_data,
        'fomc_data': fomc_data,
        'prediction_agent_output': None,
        'reflection_agent_output': None,
        'predictions_history': predictions_history.copy()
    }
    
    try:
        # Run the prediction system with retry logic
        retry_delay = INITIAL_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                final_state = bond_prediction_agent.invoke(initial_state)
                break  # Success, exit retry loop
            except Exception as api_error:
                if attempt < MAX_RETRIES - 1:
                    print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {api_error}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= BACKOFF_MULTIPLIER  # Exponential backoff
                else:
                    raise api_error  # Final attempt failed
        
        # Get actual price for this target date (if available)
        actual_price = input_.get_actual_bond_price(strips_df, target_date_str)
        
        # Store results
        prediction_result = {
            'date': current_date_str,
            'target_date': target_date_str,
            'prediction': final_state['prediction_agent_output'],
            'reflection': final_state['reflection_agent_output'],
            'actual_price': actual_price
        }
        
        all_predictions.append(prediction_result)
        predictions_history.append(prediction_result)
        
        # Keep only recent history for reflection
        if len(predictions_history) > MAX_PREDICTION_HISTORY:
            predictions_history = predictions_history[-MAX_PREDICTION_HISTORY:]
        
        print(f"   Results saved for {current_date_str}")
        
    except Exception as e:
        print(f"   ERROR: All retries failed for {current_date_str}: {e}")
        print(f"  Skipping this date and continuing...")
    
    # Move to next day
    current_date += timedelta(days=7)#1wk

# Save results to CSV
if all_predictions:
    results_df = pd.DataFrame(all_predictions)
    output_filename = f"bond_predictions_{treasury_name.replace(' ', '_')}_{model_name}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nSaved {len(all_predictions)} predictions to {output_filename}")
    print(f"Columns: {list(results_df.columns)}")
else:
    print("No predictions generated.")
