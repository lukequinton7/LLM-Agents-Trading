# A Multi-LLM-Agent Framework for Macroeconomic Signal Extraction and Financial Trading

The project addresses a novel trading framework for trading zero-coupon U.S Treasuries; it features a collaborative, two-step "Analyst-Trader" multi-agent system that makes trading decisions guided by macroeconomic analysis and time-series predictions. The system was back-tested and shown to generate significant positive returns, outperforming all benchmark models across a range of market conditions.

## Core Components

The project is built around a sequence of Python scripts that perform signal extraction and execute trades:

* **Summariser Agents (`ZH_Rapid_summariser_*.py`)**: These agents process raw financial news articles from ZeroHedge to filter and produce concise summaries related to inflation and interest rates. The performance of smaller, local models (like Gemma 3 12b) was evaluated against a ground-truth proxy (Gemini 2.5 Pro) for this task.

* **Inflation Predictor (`ZH_Inflation_Predictor_*.py`)**: A two-step Predictor/Reflector agent system that uses the generated summaries and historical data to forecast the Cleveland Nowcast Inflation Index. The framework's performance was benchmarked against a traditional ARIMA(1,1,1) model.

* **Autonomous Trading Agent (`Bond_Trade_2_Simple.py`)**: A trading agent that uses an Analyst agent's commentary and macroeconomic data to make autonomous, discretionary decisions on buying or selling U.S. Treasuries. This model was found to be susceptible to 'whipsawing' due to over-trading.

* **ARIMA Driven Trading Agent (`Bond_Trade_1_ARIMA.py`)**: A more restrictive trading agent that is heavily guided by rules based on an ensemble of ARIMA time-series price forecasts to execute a momentum-based strategy. This variant consistently delivered superior performance to the autonomous agent.

## Technologies Used

-   **Programming Language**: Python
-   **Core Libraries**:
    -   `pandas` for data manipulation.
    -   `langchain` and `langgraph` for building LLM agents.
    -   `transformers` and `huggingface_hub` for using pre-trained models.
    -   `torch` for machine learning operations.
    -   `statsmodels` for ARIMA time-series forecasting.
    -   `matplotlib` for data visualisation.
-   **Models**: Gemini 2.5 Pro, Gemma 3 series, Llama 3.1.
-   **Data Sources**: Bloomberg Terminal, Federal Reserve, and ZeroHedge.

## How to Run

The project follows a modular workflow. The core steps are:

1.  **Summarise and Filter News Articles**: Run a summariser agent to process raw news articles into summaries.
2.  **Generate Predictions**: Execute the inflation predictor script to create forward-looking macroeconomic signals.
3.  **Simulate Trades**: Run either `Bond_Trade_1_ARIMA.py` or `Bond_Trade_2_Simple.py` to backtest the trading strategies.
4.  **Analyse Results**: Use the performance and visualisation scripts to evaluate the outcomes.
