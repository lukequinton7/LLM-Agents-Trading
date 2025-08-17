import pandas as pd
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os

# Gemini Configuration
# API key (left empty for privacy)
GEMINI_API_KEY = "" 

# Configuration - data loading
csv_input_1 = 'all_tweets_aggregated.csv' 
csv_input_2 = "test tweets/+1 sample.csv"
csv_input_3 = "zerohedge_full_range_2024-05-31_to_2025-06-01.csv"  # ZeroHedge csv
csv_input_4 = "zerohedge_full_range_2024-03-31_to_2024-04-30_cleaned.csv"   # ZeroHedge csv

csv_select = csv_input_4  # dataset select

# Model choice for Gemini models
m1 = "gemini-2.5-pro" 

# Select the model to use
model_select = m1 

model_name = str(model_select).replace(":", "_").replace("/", "_").replace(".", "_") #clean string for save

max_token_limit = 8192  # Set a maximum token limit for the model
batch = True  # use batch processing (True) or single run for testing (False)

# Data Loading (DFs) - zh articles
try:
    df = pd.read_csv(csv_select)
    
    # Check if this is the ZeroHedge articles CSV format
    if 'article_body' in df.columns and 'article_headline' in df.columns:
        # For articles CSV
        df['text'] = df['article_body']  # Use article body as text
        df['created_at'] = pd.to_datetime(df['publication_datetime'], errors='coerce')
        df['headline'] = df['article_headline']
        df['url'] = df['article_url']
        df['date'] = df['created_at'].dt.date

    elif 'text' in df.columns and 'created_at' in df.columns:
        # For tweets CSV format- now redundant 
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        print("Loaded tweets CSV format")
        df['date'] = df['created_at'].dt.date
    else:
        raise ValueError("CSV must contain correct cols")
    
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()

# LangGraph State Definition
class AnalysisState(TypedDict):
    question: str
    content: str  # 'content' for articles
    result: Optional[str]
    article_info: Optional[dict]  # additional info for articles

# LLM and Prompt Setup
# ChatGoogleGenerativeAI as opposed to Ollama
model = ChatGoogleGenerativeAI(
    model=model_select,
    google_api_key=GEMINI_API_KEY,
    temperature=0.0, #temp- deterministic
    max_output_tokens=max_token_limit
)

# prompt template for articles- sentiment generation
template = '''
You are a system designed to analyse economic content and output a SINGLE numerical US Inflation sentiment score (-1, 0 or +1).
You will NEVER output text analysis. You can only output -1,0 or +1 with no additional text- this corresponds to a forward looking inflation sentiment score.
+1 means inflation is likely to increase based on your analysis of the content, -1 implies a decrease, 0 implies neutral.
The most vitally important part of this task is to NEVER output anything other than +1,0 or -1 and ALWAYS make sure you are considering inflation in particular. 

here is the question to answer: {question}

example 1: 
Content: "Inflation will be massively up next month!", "Inflation running red hot", "Prices rising faster than expected"
output (correct): +1

example 2: 
Content: "Inflation hard to predict next month- I see upward AND downward momentum", "Inflation gonna be the same as last month", "*Article unrelated to inflation*"
output (correct): 0

example 3: 
Content: "Inflation likely going down next month", "Inflation cooling big time", "I think the FED has to lower rates due to all the signs on inflation tracking down"
output (correct): -1

here is the content to analyse: {content}

Remember- you MUST only output +1, 0 or -1.
'''

# LangGraph Node Definition
def analyse_sentiment_node(state: AnalysisState) -> AnalysisState:
    """
    Analyses the content in the state and updates the state with the result.
    """
    print("---Invoking Inflation Sentiment Analysis Agent---")
    # Format the prompt using data from the state
    prompt = template.format(question=state['question'], content=state['content'])
    
    # Invoke the model with the formatted prompt
    response = model.invoke(prompt)
    
    # Gemini spicefic-  response object requires accessing the .content attribute
    state['result'] = response.content
    return state

# Graph Definition
graph_builder = StateGraph(AnalysisState)

# Add analysis function as a node named 'sentiment_analyser'
graph_builder.add_node("sentiment_analyser", analyse_sentiment_node)

# Define the entry and exit points of the graph
graph_builder.add_edge(START, "sentiment_analyser")
graph_builder.add_edge("sentiment_analyser", END)

# Compile the graph into a runnable agent
agent = graph_builder.compile()

# Main Execution Logic
# loop through articles not days/weeks
if batch:
    results = []
    if not df.empty:
        total_articles = len(df)
        print(f"Processing {total_articles} articles individually...")
        
        for index, row in df.iterrows():
            # Get article content
            article_content = str(row['text']) if pd.notna(row['text']) else ""
            
            if article_content.strip():
                # Create article info dictionary
                article_info = {
                    'index': index,
                    'date': row['date'],
                    'headline': row.get('headline', 'N/A'),
                    'url': row.get('url', 'N/A')
                }
                
                #reinforce prompt with question
                initial_state = {
                    "question": "Is inflation likely to rise (+1), remain the same (0) or fall in the near future (-1). IMPORTANT- you can ONLY output -1, 0 or +1. NEVER output any other text whatsoever (ie. max character output is 2)",
                    "content": article_content,
                    "article_info": article_info
                }
                
                final_state = agent.invoke(initial_state)
                
                # Store results with article metadata
                result_entry = {
                    "article_index": index,
                    "date": row['date'],
                    "headline": row.get('headline', 'N/A'),
                    "url": row.get('url', 'N/A'),
                    "sentiment": final_state['result']
                }
                results.append(result_entry)
                
                print(f"Article {index + 1}/{total_articles}: {final_state['result']} | {row.get('headline', 'N/A')[:60]}...")
                
            else:
                print(f"Article {index + 1}/{total_articles}: No content found, skipping.")

        # Save results after processing all articles
        if results:
            results_df = pd.DataFrame(results)
            output_filename = f"article_inflation_results_{model_name}.csv"
            results_df.to_csv(output_filename, index=False)
            print(f"\nSaved {len(results)} article sentiment results to {output_filename}")
            
            # Print summary statistics
            sentiment_counts = results_df['sentiment'].value_counts()
            print(f"\nSentiment Summary:")
            print(f"Positive (+1): {sentiment_counts.get('+1', 0)}")
            print(f"Neutral (0): {sentiment_counts.get('0', 0)}")
            print(f"Negative (-1): {sentiment_counts.get('-1', 0)}")
        else:
            print("No articles were processed successfully.")
    else:
        print("No article data was loaded, LLM was not invoked.")

# If not in batch mode, we run the agent multiple times with the same data for testing
else:

    print(f"model temperature: (>0 non-deterministic) {model.temperature}")
    
    if not df.empty:
        # Use the first article for testing
        first_article = df.iloc[0]
        article_content = str(first_article['text']) if pd.notna(first_article['text']) else ""
        
        if article_content.strip():
            print(f"Testing with first article: {first_article.get('headline', 'N/A')[:60]}...")
            
            for i in range(0, 10):
                initial_state = {
                    "question": "Is inflation likely to rise (+1), remain the same (0) or fall in the near future (-1). IMPORTANT- you can ONLY output -1, 0 or +1. NEVER output any other text whatsoever (ie. max character output is 2)",
                    "content": article_content,
                }
                final_state = agent.invoke(initial_state)
                print(f"Run {i+1}: {final_state['result']}")
        else:
            print("First article has no content to analyse.")
    else:
        print("No article data was loaded, LLM was not invoked.")