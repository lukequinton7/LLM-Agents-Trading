import pandas as pd
from typing import TypedDict, Optional
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
from datetime import timedelta

# Configuration 
summaries_csv_file = "ZH_article_summaries_inflation_gemma3_12b.csv" #zh summaries from summariser agent
nowcast_csv_file = "Nowcast historic.csv" #historic inflation data
model_select = "gemma3:12b" 
model_name = str(model_select).replace(":", "_").replace("/", "_") #change string for save
max_token_limit = 16000
TASK_TYPE = "inflation"

# Data Loading- to df
def load_summaries():
    summaries_df = pd.read_csv(summaries_csv_file)
    summaries_df['datetime'] = pd.to_datetime(summaries_df['datetime'])
    return summaries_df

def load_nowcast_data():
    nowcast_df = pd.read_csv(nowcast_csv_file)
    nowcast_df['Date'] = pd.to_datetime(nowcast_df['Date'])
    nowcast_df = nowcast_df.sort_values('Date', ascending=False).reset_index(drop=True)
    return nowcast_df

def get_historical_nowcast(nowcast_df, reference_date, days_back=100):
    reference_date = pd.to_datetime(reference_date)
    start_date = reference_date - timedelta(days=days_back)
    
    historical_data = nowcast_df[
        (nowcast_df['Date'] >= start_date) & 
        (nowcast_df['Date'] <= reference_date)
    ].sort_values('Date')
    
    if historical_data.empty:
        return "No nowcast data available"
    
    nowcast_text = f"Historical Nowcast Inflation Data (last {len(historical_data)} readings):\n\n"
    for _, row in historical_data.iterrows():
        nowcast_text += f"{row['Date'].strftime('%Y-%m-%d')}: {row['Nowcast']}%\n"
    
    if len(historical_data) >= 2:
        latest_value = historical_data.iloc[-1]['Nowcast']
        oldest_value = historical_data.iloc[0]['Nowcast']
        change = latest_value - oldest_value
        nowcast_text += f"\nTrend: {change:+.2f}% (from {oldest_value}% to {latest_value}%)"
    
    return nowcast_text

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
    prediction_result: Optional[str]

# Model Setup - ollama
model = OllamaLLM(model=model_select, temperature=0.0, num_predict=max_token_limit) #deterministic

inflation_predictor_template = '''
You are an inflation prediction agent designed to receive relevant inputs and predict inflation in 14 days time (particurlarly Nowcast, Bloomberg ticker: CLEVINF Index).

You will use present and historical nowcast inflation data and also inflation news article summaries to make your prediction.

You will have access to your previous predictions so you can recalibrate your model based on past performance

You must output a single numerical value for your prediction, which is the expected nowcast inflation rate in 14 days time.

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

Here are the relevant news article summaries from the last 3 days.....
{summary_context}

'''

inflation_reflection_template = '''
You are an inflation prediction agent designed to reflect on another agents predictions and improve the output.

The previous agent has predicted nowcast inflation for 14 days in the future.

You must compare the previous agent's predictions with the actual nowcast inflation data (matching prediction for date with realised nowcast date).

eg. prediction date: 2023-01-01, prediction for 2023-15-01: 3.70, compared to actual nowcast inflation for 2023-15-01: 3.50...

You must output a single numerical value for your prediction, which is the expected nowcast inflation rate in 14 days time.

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

Here are the previous agent's predictions
{previous_predictions}

If there is not yet any nowcast inflation data available, you must output previous agents prediction (unchanged). 

'''





def predictor_node(state: PredictorState) -> PredictorState:
    prompt = inflation_predictor_template.format(
        summary_context=state['summary_context'], 
        nowcast_data=state['nowcast_data']
    )
    
    response = model.invoke(prompt)
    state['prediction_result'] = str(response).strip()
    return state

# Build graph
predictor_graph_builder = StateGraph(PredictorState)
predictor_graph_builder.add_node("predictor", predictor_node)
predictor_graph_builder.add_edge(START, "predictor")
predictor_graph_builder.add_edge("predictor", END)
predictor_agent = predictor_graph_builder.compile()

# Main Processing logic
all_predictions = []
unique_dates = sorted(summaries_df['datetime'].dt.date.unique())

for current_date in unique_dates[3:]:  # Start from 4th date- giving ample previous days text data
    print(f"Processing: {current_date}")
    
    # Get 3-day window
    start_date = pd.to_datetime(current_date) - timedelta(days=3)
    end_date = pd.to_datetime(current_date) - timedelta(days=1)
    
    # Filter relevant summaries
    all_summaries_in_window = summaries_df[
        (summaries_df['datetime'].dt.date >= start_date.date()) &
        (summaries_df['datetime'].dt.date <= end_date.date())
    ]
    
    relevant_summaries_df = all_summaries_in_window[
        ~all_summaries_in_window['summary'].str.contains('no relevant content', case=False, na=False) #string matching- string ANYWHERE
    ]
    
    print(f"  Date window: {start_date.date()} to {end_date.date()}")
    print(f"  Total articles in window: {len(all_summaries_in_window)}")
    print(f"  Relevant articles (after filtering): {len(relevant_summaries_df)}")
    
    if len(relevant_summaries_df) == 0:
        continue
    
    # Truncate each individual summary to 200 words but keep all relevant ones
    summaries_list = relevant_summaries_df['summary'].astype(str).tolist()
    truncated_summaries = truncate_individual_summaries(summaries_list, max_words_per_summary=200)
    summary_context = "\n\n---\n\n".join(truncated_summaries)
    
    print(f"  All {len(truncated_summaries)} relevant summaries included (each capped at 200 words)")
    
    # Get nowcast data
    nowcast_data = get_historical_nowcast(nowcast_df, end_date, days_back=100)
    
    # Make prediction
    initial_state = {
        "summary_context": summary_context,
        "nowcast_data": nowcast_data,
        "prediction_result": None
    }
    
    try:
        final_state = predictor_agent.invoke(initial_state)
        prediction_entry = {
            "prediction_date": current_date,
            "prediction_30_day_fwd": final_state['prediction_result'],
            "num_summaries_used": len(truncated_summaries)
        }
        all_predictions.append(prediction_entry)
        print(f"Prediction: {final_state['prediction_result']}")
        
    except Exception as e:
        print(f"Error: {e}")

# Save results
if all_predictions:
    predictions_df = pd.DataFrame(all_predictions)
    output_filename = f"daily_predictions_{TASK_TYPE}_{model_name}.csv"
    predictions_df.to_csv(output_filename, index=False)
    print(f"\nSaved {len(all_predictions)} predictions to {output_filename}")
else:
    print("No predictions generated.")



    