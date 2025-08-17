import pandas as pd


# Read the CSV files containing daily inflation results for different models INFLATION
df_gemma = pd.read_csv("daily_inflation_results_gemma3_12b.csv")    

df_deepseek = pd.read_csv("daily_inflation_results_deepseek-r1_14b.csv")

df_qwen = pd.read_csv("daily_inflation_results_qwen3_14b.csv")

df_llama = pd.read_csv("daily_inflation_results_llama3.1_8b.csv")

# Read the CSV files containing daily inflation results for different models RATES
df_gemma_rates = pd.read_csv("daily_rates_results_gemma3_12b.csv")   

df_deepseek_rates = pd.read_csv("daily_rates_results_deepseek-r1_14b.csv")

df_qwen_rates = pd.read_csv("daily_rates_results_qwen3_14b.csv")

df_llama_rates = pd.read_csv("daily_rates_results_llama3.1_8b.csv")






def clean_sentiment_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame by extracting the numerical sentiment score
    and ensuring correct data types for 'date' and 'sentiment' columns.

    Args:
        daily_df (pd.DataFrame): The raw DataFrame with 'date' and verbose 'sentiment' columns.

    Returns:
        pd.DataFrame: A cleaned DataFrame with a numerical 'sentiment' column 
                      and a datetime 'date' column.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = daily_df.copy()

    # Extract the final integer from the verbose sentiment string using a regular expression.
    df['sentiment'] = df['sentiment'].astype(str).str.extract(r'(-?\d+)$', expand=False)
    
    # Convert the extracted column to a numeric type.
    # 'coerce' will replace any non-numeric values with NaN (Not a Number).
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')

    # Convert the 'date' column to a datetime object.
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def aggregate_sentiment_by_month(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a cleaned DataFrame of daily sentiment data to a monthly sum.

    Args:
        cleaned_df (pd.DataFrame): A DataFrame with a numerical 'sentiment' column
                                   and a datetime 'date' column.

    Returns:
        pd.DataFrame: A new DataFrame with sentiment summed by month.
    """
    # Create a copy to avoid modifying the input DataFrame
    df = cleaned_df.copy()

    # Set the date as the index to enable time-series resampling.
    df.set_index('date', inplace=True)

    # Resample the data by month start ('MS') and sum the sentiment values.
    # .sum() will ignore NaN values by default.
    monthly_sentiment = df['sentiment'].resample('MS').sum()

    # Convert the resulting Series back into a DataFrame.
    monthly_sentiment_df = monthly_sentiment.to_frame()

    # Format the index to show only the year and month (e.g., '2024-06').
    monthly_sentiment_df.index = monthly_sentiment_df.index.strftime('%Y-%m')
    
    return monthly_sentiment_df


#clean data sets 
df_deepseek_clean = clean_sentiment_data(df_deepseek)

df_gemma_clean = clean_sentiment_data(df_gemma)

df_qwen_clean = clean_sentiment_data(df_qwen)

df_llama_clean = clean_sentiment_data(df_llama)


df_deepseek_rates_clean = clean_sentiment_data(df_deepseek_rates)

df_gemma_rates_clean = clean_sentiment_data(df_gemma_rates)

df_qwen_rates_clean = clean_sentiment_data(df_qwen_rates)

df_llama_rates_clean = clean_sentiment_data(df_llama_rates)


#aggregarte clean data sets by month
#inflation
df_deepseek_mth = aggregate_sentiment_by_month(df_deepseek_clean)
print("Monthly Inflation Sentiment for DeepSeek:")
print(df_deepseek_mth)

df_gemma_mth = aggregate_sentiment_by_month(df_gemma_clean)
print("Monthly Inflation Sentiment for Gemma:")
print(df_gemma_mth)

df_qwen_mth = aggregate_sentiment_by_month(df_qwen_clean)
print("Monthly Inflation Sentiment for Qwen:")
print(df_qwen_mth)  

df_llama_mth = aggregate_sentiment_by_month(df_llama_clean)
print("Monthly Inflation Sentiment for Llama:")
print(df_llama_mth)


#rates
df_deepseek_rates_mth = aggregate_sentiment_by_month(df_deepseek_rates_clean)
print("Monthly Rates Sentiment for DeepSeek:")
print(df_deepseek_rates_mth)

df_gemma_rates_mth = aggregate_sentiment_by_month(df_gemma_rates_clean)
print("Monthly Rates Sentiment for Gemma:")
print(df_gemma_rates_mth)

df_qwen_rates_mth = aggregate_sentiment_by_month(df_qwen_rates_clean)
print("Monthly Rates Sentiment for Qwen:")  
print(df_qwen_rates_mth)

df_llama_rates_mth = aggregate_sentiment_by_month(df_llama_rates_clean)
print("Monthly Rates Sentiment for Llama:")
print(df_llama_rates_mth)


