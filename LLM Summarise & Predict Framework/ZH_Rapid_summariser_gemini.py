import pandas as pd
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os

# Gemini Configuration- API
GEMINI_API_KEY = "" #hidden as personal data

# Configuration - data loading
csv_input_1 = "zerohedge_full_range_2024-05-31_to_2025-06-01.csv"  # ZeroHedge csv
csv_input_2 = "zerohedge_full_range_2024-03-31_to_2024-04-30_cleaned.csv"  # ZeroHedge csv
csv_input_3 = "zerohedge_full_range_2024-03-31_to_2024-04-30_cleaned_snipped.csv"  # ZeroHedge csv

csv_select = csv_input_2  # choose dataset

# Model choice - now using Gemini model (will work for other Gemini variants)
m1 = "gemini-2.5-pro" 

# Select the model to use
model_select = m1 

model_name = str(model_select).replace(":", "_").replace("/", "_").replace(".", "_") #string change for save

max_token_limit = 12000  # Set a maximum token limit to prevent wasting credits


### IMPORTANT- IMPORTANT -IMPORTANT - IMPORTANT ###         

temp_type = "rates"  #for save


# Data Loading -zh articles
try:
    df = pd.read_csv(csv_select)
    
    # Check if this is the ZeroHedge articles CSV format
    if 'article_body' in df.columns and 'article_headline' in df.columns:
        # For articles CSV
        df['text'] = df['article_body']  # Use article body as text
        df['created_at'] = pd.to_datetime(df['publication_datetime'], errors='coerce')
        df['headline'] = df['article_headline']
        df['url'] = df['article_url']
        print("Loaded articles CSV format")

    else:
        raise ValueError("CSV must contain either correct cols")
    
except FileNotFoundError:
    print(f"Error: The file '{csv_select}' was not found. Please check the file path.")
    df = pd.DataFrame() # Create an empty DataFrame to continue
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()


#LangGraph State Definition 
class AnalysisState(TypedDict):
    content: str  # article content
    result: Optional[str]
    article_info: Optional[dict]  # additional info for articles

#LLM and Prompt Setup 
# using ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(
    model=model_select,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0, # Lower temperature for deterministic outputs
    max_output_tokens=max_token_limit
)

# inflation 
inflation_template = '''
You are an economic summary generator designed to output a summary on INFLATION content that is no greater than 150 words.

Your task is to analyse the below article in this prompt and provide a concise summary of the key points related to INFLATION, particularly US inflation.

IMPORTANT: if the article is not relevant to INFLATION, you should output "No relevant content" and nothing else.

You should pay close attention to any references to inflation rates, interest rates, CPI, forecasts, price rises and any economic indicators that suggest how inflation is
expected to change in the near future with numerical values where possible.

Remember- your output should not exceed 150 words and should focus on the topic of INFLATION.

If you conclude the article is not relevant to INFLATION, you should output "No relevant content" and nothing else.

Here is the article to summarise.....

{content}
'''

#rates template
rates_template = '''
You are an economic summary generator designed to output a summary on US INTEREST RATES content that is no greater than 150 words.

Your task is to analyse the below article in this prompt and provide a concise summary of the key points related to US Rates, particularly FED decisions.

IMPORTANT: if the article is not relevant to rates, you should output "No relevant content" and nothing else.

You should pay close attention to any references to interest rates, The FED, forecasts, yields and any economic indicators that suggest how US rates
expected to change in the near future with numerical values where possible.

Remember- your output should not exceed 150 words and should focus on the topic of INTEREST RATES.

If you conclude the article is not relevant to INTEREST RATES, you should output "No relevant content" and nothing else.

Here is the article to summarise.....

{content}
'''


#template choice
template_select = rates_template  # Change this to rates_template for rates instead of inflation


# LangGraph Node Definition
def analyse_sentiment_node(state: AnalysisState) -> AnalysisState:
    """
    Analyses the content in the state and updates the state with the result.
    """
    print("---Invoking Inflation Summary Agent---")
    prompt = template_select.format(content=state['content'])
    
    # invoke the model with the formatted prompt
    response = model.invoke(prompt)
    
    # Gemini response object requires accessing the .content attribute (ollama different)
    state['result'] = response.content
    return state

# Graph Definition
graph_builder = StateGraph(AnalysisState)

# add analysis function as a node named 'sentiment_analyser'
graph_builder.add_node("sentiment_analyser", analyse_sentiment_node)

# define the entry and exit points of the graph 
graph_builder.add_edge(START, "sentiment_analyser")
graph_builder.add_edge("sentiment_analyser", END)

# compile the graph into a runnable agent
agent = graph_builder.compile()

# Main Execution Logic
# loop through articles (not dates)

results = []
if not df.empty:
    total_articles = len(df)
    print(f"Processing {total_articles} articles individually...")
    
    for index, row in df.iterrows():
        # Get article content
        article_content = str(row['text']) if pd.notna(row['text']) else ""
        
        if article_content.strip():
            # create article info dictionary
            article_info = {
                'index': index,
                'datetime': row['created_at'],
                'headline': row.get('headline', 'N/A'),
                'url': row.get('url', 'N/A')
            }
            
            initial_state = {
                "content": article_content,
                "article_info": article_info
            }
            
            final_state = agent.invoke(initial_state)
            
            # result_entry contains only date, title and summary.
            result_entry = {
                "datetime": row['created_at'],
                "headline": row.get('headline', 'N/A'),
                "summary": final_state['result']
            }
            results.append(result_entry)
            
            print(f"Article {index + 1}/{total_articles}: Summary generated for {row.get('headline', 'N/A')[:60]}...") #capped headline output
            print("")
            print(final_state['result']) #summary
            print("")

            
        else:
            print(f"Article {index + 1}/{total_articles}: No content found, skipping.")

    # Save results after processing all articles
    if results:
        results_df = pd.DataFrame(results)
        output_filename = f"ZH_article_summaries_{temp_type}_{model_name}.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"\nSaved {len(results)} article summaries to {output_filename}")
        
    else:
        print("No articles were processed successfully.")
