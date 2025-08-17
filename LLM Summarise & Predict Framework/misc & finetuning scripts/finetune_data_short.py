import pandas as pd
from transformers import AutoTokenizer


csv_article = "zerohedge_full_range_2024-03-31_to_2024-04-30_cleaned.csv" 
csv_inflation = "article_inflation_results_gemini-2_5-pro.csv"
csv_rates = "article_rates_results_gemini-2_5-pro.csv"




#finetune template prompts

inflation_template = '''
You are a system designed to analyse economic content and output a SINGLE numerical US Inflation sentiment score (-1, 0 or +1).
You will NEVER output text analysis. You can only output -1,0 or +1 with no additional text- this corresponds to a forward looking inflation sentiment score.
+1 means inflation is likely to increase based on your analysis of the content, -1 implies a decrease, 0 implies neutral.
The most vitally important part of this task is to NEVER output anything other than +1,0 or -1 and ALWAYS make sure you are considering inflation in particular. 

Remember- you MUST only output +1, 0 or -1.

here is the article.....
'''


rates_template = '''
You are a system designed to analyse economic content and output a SINGLE numerical Federal Funds Target Range - Lower Limit sentiment score (-1, 0 or +1).
You will NEVER output text analysis. You can only output -1,0 or +1 with no additional text- this corresponds to a forward looking FFTR sentiment score.
+1 means the FED is likely to increase rates based on your analysis of the article at the next FOMC meeting, -1 implies a decrease, 0 implies no change.
The most vitally important part of this task is to NEVER output anything other than +1,0 or -1 and ALWAYS make sure you are considering FFTR in particular.  

Remember- you MUST only output +1, 0 or -1.

here is the article.....
'''





#combine gemini silver standard results df with prompts to create a finetune dataset

def make_finetune_data(csv_target, temp):
    """    
    Function to create a finetune dataset from a CSV file containing articles and their sentiment scores
    """


    #extracts all data to df from csv
    try:
        df_articles = pd.read_csv(csv_article)

        df_target = pd.read_csv(csv_target)

        # insert a new column that inserts the template above in to each row
        df_articles['prompt_pt1'] = temp

        #concatenate the prompt_pt1 and article_body columns ato create a new column called 'prompt'
        df_articles['prompt'] = df_articles['prompt_pt1'] + df_articles['article_body']

        #remove the prompt_pt1 column, article_url and the article_body column
        df_articles.drop(columns=['prompt_pt1', 'article_body','article_url'], inplace=True)

        #add cols from df_inflation to df_articles
        df_articles['sentiment'] = df_target['sentiment']

        #clean rates and inflation sentiment- convert to text then if output 1 we make +1
        
        if temp == inflation_template or temp == rates_template:
            df_articles['sentiment'] = df_articles['sentiment'].replace({1: '+1', 0: '0', -1: '-1'})
        df_articles['sentiment'] = df_articles['sentiment'].astype(str)


        print(df_articles.tail(40))  # Display the first few rows of the DataFrame for verification

        return df_articles

    except Exception as e:
        print(f"Error loading CSV: {e}")


# Load the finetune data for inflation and rates and conflict (MISSING)
df_inflation = make_finetune_data(csv_inflation, inflation_template)
df_rates = make_finetune_data(csv_rates, rates_template)

#stack the two dataframes on top of each other
df_finetune = pd.concat([df_inflation, df_rates], ignore_index=True)



#cap the prompt to approcimately the equivalnt of 1500 tokens max (as llms perceive them)
#df_finetune['prompt'] = df_finetune['prompt'].str.slice(0, 4500)




# Load the same tokenizer for training
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

MAX_PROMPT_TOKENS = 1100

def truncate_by_tokens(text, max_length):
    """
    Truncates text to a maximum token length, respecting token boundaries.
    """
    # Tokenize the text without adding special tokens
    input_ids = tokenizer(text, add_special_tokens=False).input_ids
    
    # If the text is already short enough, return it as is
    if len(input_ids) <= max_length:
        return text
    
    # Otherwise, truncate the token list and decode back to a string
    truncated_ids = input_ids[:max_length]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

# --- How to apply it in one line to your DataFrame ---

# Assuming 'df' is your pandas DataFrame
# This will apply the function to every row in the 'prompt' column
df_finetune['prompt'] = df_finetune['prompt'].apply(lambda x: truncate_by_tokens(x, MAX_PROMPT_TOKENS))





df_finetune.head(5)
df_finetune.tail(5)



#RETURN THE WHOLE ENTRY FROM DF THAT HAS PROMPT WITH MAX CARS USING ILOC
max_prompt_entry = df_finetune.loc[df_finetune['prompt'].str.len().idxmax()]
print("Entry with max characters in prompt:")               
print(max_prompt_entry)



#print the first prompt and last prompt
print("First prompt:", df_finetune['prompt'].iloc[0])
print("Last prompt:", df_finetune['prompt'].iloc[-1])

#convert the DataFrame to a CSV file
df_finetune.to_csv("finetune_data_short.csv", index=False)

#clean rates and inflation sentiment- if output 1 we make +1 in csv






