import pandas as pd
from typing import TypedDict, Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from datetime import timedelta, datetime
import clean_input_df as input_ #df cleaning and utility functions
import time

# Configuration
GEMINI_API_KEY = "" #blank for privacy
summaries_csv_file = "ZH_article_summaries_full_inflation_gemma3_12b.csv" #gemma 3 agent summaries (zh articles)
nowcast_csv_file = "Nowcast historic.csv" #historic inflation data
model_select = "gemini-2.5-pro"
model_name = str(model_select).replace(":", "_").replace("/", "_").replace(".", "_")
max_token_limit = 20000
MAX_TRADE_HISTORY = 21  # Number of previous trades to keep for reflection
HISTORIC_DATA_DAYS = 30  # Number of days of historic data to use for rates and inflation
HISTORIC_DATA_DAYS_BOND = 100  # Number of days of historic data to use for rates and inflation
HISTORIC_DATA_DAYS_STATEMENT = 50 #days worth of fomc statement data 
treasury_name = "SP 0 05 15 50" #25yr

ABLATION_NO_SUMMARIES = True #flag for ablation- or set HISTORIC_DAYS to 0

# Manual date range configuration
START_DATE = "2024-09-04"  #  start date
END_DATE = "2024-11-04"    # end date

# robustness configs
MAX_RETRIES = 3  # API retry attempts
INITIAL_RETRY_DELAY = 5  # initial retry delay in seconds
BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

#load cleaned historic dataframes
strips_df = input_.STRIPS_to_df() #historic STRIPS data
vol_df = input_.vol_df_30_day(strips_df)
rates_df = input_.EFFR_to_df()  #historic rate data
nowcast_df = input_.Nowcast_to_df() #historic nowcast data
FOMC_df = input_.FOMC_PDFs_to_df() #historic FOMC statements
yc_df = input_.yields_to_df() #historic YC data
arima_df = input_.ARIMA_to_df(treasury_name)

# LangGraph State Definition
class TradingState(TypedDict):
    current_date: str
    strips_data: str
    vol_data: str
    rates_data: str
    nowcast_data: str
    yc_data: str
    fomc_data: str
    current_position: float
    current_price: float
    ARIMA_predictions: str
    analyst_output: Optional[str]
    trader_output: Optional[str]
    trade_history: List[Dict]

# Model Setup - gemini specific (works for other gemini models)
model = ChatGoogleGenerativeAI(
    model=model_select,
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,    #CHANGED FOR REPEAT TRIALS
    max_output_tokens=max_token_limit
)

# Agent Templates

#analyst prompt
bond_analyst_template = """
Today is {current_date}.

You are a bond analyst agent designed to craft 3 short paragraphs to give a trading agent insight into: {treasury_name}
and related macroeconomic phenomena as well as the agent's previous decision making.

Guide:
- Do not explicitly advise trades, rather provide colour
- Attempt to detect regime changes or macro shocks

Paragraph 1: 
Paragraph one has the heading: Quantitative Macroeconomic Insights. You will use the following numerical data to craft it:

Here is the previous Treasury price and yield data:
{strips_data}

Here is the previous 30 day daily price volatility:
{vol_data}

Here is the previous FED rates data:
{rates_data}

Here is the previous nowcast inflation data:
{nowcast_data}

Here is the previous term structure (yield curve) data:
{yc_data}

Paragraph 2: 
Paragraph two has the heading: FOMC Statement Analysis. You will use the following text data to craft it:

Here is the most recent FOMC statements:
{fomc_data}

Paragraph 3:
Paragraph three has the heading: Previous Trade Reflection. You will attempt to give the trading agent insight on previous recent 
mistakes it made with quantitative commentary. If no data do not comment

Here are previous trades:
{previous_trades}
"""

#trade agent
bond_trader_template = """
As an advanced trading strategy agent for {treasury_name} zero coupon bond, analyze the following data to formulate an opportunistic
trading decision:

Current Date: {current_date}
Current Portfolio:
- Position (-1.0 to 1.0): {current_position}

Bond Price: ${current_price:.2f}

ARIMA price forecasts (IMPORTANT): {ARIMA_predictions}

Analyst Agent Commentary:
{analyst_commentary}.....

Previous Trades:
{previous_trades}

Consider the above and the guide + rules below to make your decision:

Guide:
- You can go short or long; consider the state of the current portfolio
- We expect 10-30 trades per year, so carefully seek optimal trade moments
- Try not to over-trade; you can exit positions early but this should not be a common occurance

Rules:
- IMPORTANT: You cannot recommend trades that push overall position above 1.0 or below -1.0  (e.g. can't BUY if you are already long 1.0)
- Hold positions for 7+ days where possible

- Recommend BUY if:
  a) All ARIMA horizon predictions are firmly ABOVE current bond price (1.0) OR
     Short and mid-term ARIMA predictions are firmly ABOVE current price (0.5)
     AND analyst commentary not likely to severely negatively impact bonds.
     If you are already in a short position you will need more compelling evidence to
     exit the trade early.

- Recommend SELL if:
  a) All ARIMA horizon predictions are firmly BELOW current bond price (1.0) OR
     Short and mid-term ARIMA predictions are firmly BELOW current price (0.50)
     AND analyst commentary not likely to severely positively impact bonds.
     If you are already in a long position you will need more compelling evidence to
     exit the trade early.

- Recommend HOLD if:
  a) You are long and upward/neutral ARIMA trend continuing without significant reversal signs
  b) You are short and downward/neutral ARIMA trend continuing without significant reversal signs
  c) You have no position and future direction is unclear according to ARIMA and other indicators

Provide your trading strategy in the following format:
Recommendation: [BUY/SELL/HOLD]
Trade Size: [0.0, 0.50 or 1.0] (0 if HOLD)
Explanation: [<100 word rationale for the decision] (ensure response starts on same line)

(formatting notes: if BUY/SELL must always be trade size 0.5 or 1.0,
if HOLD, always trade size 0)

"""

# Agent Nodes
def analyst_agent_node(state: TradingState) -> TradingState:
    # Format previous trades history
    trades_data = "No previous trades available."
    if state['trade_history']:
        formatted_trades = []
        for trade in state['trade_history'][-MAX_TRADE_HISTORY:]:
            formatted_trades.append(
                f"Date: {trade['date']}, Action: {trade['recommendation']}, "
                f"Size: {trade['position_size']}, Price: ${trade['price']:.2f}, "
                f"Explanation: {trade['explanation']}"
            )
        trades_data = "\n".join(formatted_trades)
    
    prompt = bond_analyst_template.format(
        current_date=state['current_date'],
        treasury_name=treasury_name,
        strips_data=state['strips_data'],
        vol_data=state['vol_data'],
        rates_data=state['rates_data'],
        nowcast_data=state['nowcast_data'],
        yc_data=state['yc_data'],
        fomc_data=state['fomc_data'],
        previous_trades=trades_data
    )
    
    response = model.invoke(prompt)
    state['analyst_output'] = str(response.content).strip()
    print(f"  Analyst Agent: {state['analyst_output'][:10000]}...") #cap just in case ignores prompt
    return state

def trader_agent_node(state: TradingState) -> TradingState:
    # Format previous trades for trader
    trades_data = "No previous trades available."
    if state['trade_history']:
        formatted_trades = []
        for trade in state['trade_history'][-5:]:  # Last 5 trades for context
            formatted_trades.append(
                f"Date: {trade['date']}, Action: {trade['recommendation']}, "
                f"Size: {trade['position_size']}, Price: ${trade['price']:.2f}"
            )
        trades_data = "\n".join(formatted_trades)
    
    prompt = bond_trader_template.format(
        treasury_name=treasury_name,
        current_date=state['current_date'],
        current_position=state['current_position'],
        current_price=state['current_price'],
        ARIMA_predictions=state['ARIMA_predictions'],
        analyst_commentary=state['analyst_output'],
        previous_trades=trades_data
    )
    
    response = model.invoke(prompt)
    state['trader_output'] = str(response.content).strip()
    print(f"  → Trader Agent: {state['trader_output'][:10000]}...")
    return state

# Build graph
graph_builder = StateGraph(TradingState)
graph_builder.add_node("analyst", analyst_agent_node)
graph_builder.add_node("trader", trader_agent_node)

graph_builder.add_edge(START, "analyst")
graph_builder.add_edge("analyst", "trader")
graph_builder.add_edge("trader", END)

bond_trading_agent = graph_builder.compile()

# Portfolio Management Functions
def parse_trader_output(trader_output):
    """parse the structured output from the trader agent"""
    #default
    lines = trader_output.split('\n')
    recommendation = "HOLD"
    position_size = 0.0
    explanation = "No explanation provided"
    
    #separate out text outputs based on start of line
    for line in lines:
        if line.startswith("Recommendation:"):
            recommendation = line.split(":", 1)[1].strip()
        elif line.startswith("Trade Size:"):
            try:
                position_size = float(line.split(":", 1)[1].strip())
            except:
                position_size = 0.0
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()
    
    return recommendation, position_size, explanation

# Main Processing Loop- daily simulation
all_trades = []
trade_history = []
current_position = 0.0  # Portfolio position (-1.0 to 1.0)

# Generate date range
start_date = pd.to_datetime(START_DATE)
end_date = pd.to_datetime(END_DATE)
current_date = start_date

print(f"Starting bond trading system from {START_DATE} to {END_DATE}")
print(f"Treasury: {treasury_name}")
print()
print()

while current_date <= end_date:
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    print(f"Processing: {current_date_str}")
    
    # Get current bond price
    current_price = input_.get_actual_bond_price(strips_df, current_date_str)
    if current_price is None:
        print(f"  No price data for {current_date_str}, skip.")
        current_date += timedelta(days=1)
        continue
    
    # Get ARIMA predictions (just for current date- not past dates)  
    arima_data = input_.get_last_n_days_data(arima_df, current_date_str, 1)
    
    # Get last n days of data from each source
    strips_data = input_.get_last_n_days_data(strips_df, current_date_str, HISTORIC_DATA_DAYS_BOND)
    vol_data = input_.get_last_n_days_data(vol_df, current_date_str, HISTORIC_DATA_DAYS)
    rates_data = input_.get_last_n_days_data(rates_df, current_date_str, HISTORIC_DATA_DAYS)
    nowcast_data = input_.get_last_n_days_data(nowcast_df, current_date_str, HISTORIC_DATA_DAYS)
    yc_data = input_.get_last_n_days_data(yc_df, current_date_str, HISTORIC_DATA_DAYS)
    fomc_data = input_.get_last_n_days_data(FOMC_df, current_date_str, HISTORIC_DATA_DAYS_STATEMENT)
    
    # Prepare state
    initial_state = {
        'current_date': current_date_str,
        'strips_data': strips_data,
        'vol_data': vol_data,
        'rates_data': rates_data,
        'nowcast_data': nowcast_data,
        'yc_data': yc_data,
        'fomc_data': fomc_data,
        'current_position': current_position,
        'current_price': current_price,
        'ARIMA_predictions': arima_data,
        'analyst_output': None,
        'trader_output': None,
        'trade_history': trade_history.copy()
    }
    
    try:
        # Run the trading system with retry logic- api fails common
        retry_delay = INITIAL_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                final_state = bond_trading_agent.invoke(initial_state)
                break  # Success, exit retry loop
            except Exception as api_error:
                if attempt < MAX_RETRIES - 1:
                    print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {api_error}")

                    time.sleep(retry_delay)
                    retry_delay *= BACKOFF_MULTIPLIER
                else:
                    raise api_error
        
        # Parse trader output- get new variables
        recommendation, position_size, explanation = parse_trader_output(final_state['trader_output'])
        
        # Execute trade logic
        trade_executed = False
        if recommendation == "BUY" and current_position < 1.0:
            # Update portfolio position- no action if too long
            current_position = min(1.0, current_position + position_size)
            trade_executed = True
            
        elif recommendation == "SELL" and current_position > -1.0:
            # update portfolio position
            current_position = max(-1.0, current_position - position_size)
            trade_executed = True
        
        # Store trade result
        trade_result = {
            'date': current_date_str,
            'recommendation': recommendation,
            'position_size': position_size,
            'price': current_price,
            'new_position': current_position,
            'explanation': explanation,
            'analyst_commentary': final_state['analyst_output'],
            'trader_full_output': final_state['trader_output'],
            'trade_executed': trade_executed
        }
        
        all_trades.append(trade_result)
        
        # Add to trade history only if trade was executed
        if trade_executed:
            trade_history.append(trade_result)
            # Keep only recent history
            if len(trade_history) > MAX_TRADE_HISTORY:
                trade_history = trade_history[-MAX_TRADE_HISTORY:]
        
        print(f" {recommendation} | Position: {current_position:.2f} | Price: ${current_price:.2f}")
        if trade_executed:
            print(f"   TRADE EXECUTED: {explanation[:50]}...")
        print()
        print()

        # Realised P&L when flat- print for output to assess P&L
        if current_position == 0 and all_trades:
            realized_pnl = sum(t['position_size'] * t['price'] if t['recommendation'] == 'SELL' 
                              else -t['position_size'] * t['price'] for t in all_trades if t['trade_executed'])
            print(f"  Realised P&L: ${realized_pnl:.2f}")
        
        print()
        print()
        print()



    except Exception as e:
        print(f"   ERROR: All retries failed for {current_date_str}: {e}")
        print(f"  Skipping this date and continuing...")
        print()
        print()
    
    # Move to next day (skip weekends)
    if current_date.weekday() == 4:  # Friday - jump to Monday
        current_date += timedelta(days=3)
    else:
        current_date += timedelta(days=1)

# save results to CSV
if all_trades:
    results_df = pd.DataFrame(all_trades)
    output_filename = f"bond_trades_{treasury_name.replace(' ', '_')}_{model_name}_TRIAL_5.csv"
    results_df.to_csv(output_filename, index=False)
    
    # Calculate trading statistics
    executed_trades = len([t for t in all_trades if t['trade_executed']])
    print(f"\nTrading Summary:")
    print(f"Total decisions: {len(all_trades)}")
    print(f"Trades executed: {executed_trades}")
    print(f"Final position: {current_position:.2f}")
else:
    print("No trades generated")






