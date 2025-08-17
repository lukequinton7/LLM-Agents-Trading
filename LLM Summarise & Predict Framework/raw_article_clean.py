import pandas as pd
import re
import os

finetuning_csv = "zerohedge_full_range_2024-03-31_to_2024-04-30.csv"  
main_csv = "zerohedge_full_range_2024-05-31_to_2025-06-01.csv"  



def clean_csv(input_file):
    """Clean ZeroHedge CSV by removing footer and emojis, save as _cleaned.csv"""
    
    # Load CSV
    df = pd.read_csv(input_file)
    
    # Define cleaning patterns
    footer_pattern = r"Assistance and Requests:Click here.*?Notice on Racial Discrimination\."
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]+')
    
    # Clean article_body
    df['article_body'] = df['article_body'].astype(str).apply(lambda x: 
        emoji_pattern.sub('', re.sub(footer_pattern, '', x, flags=re.DOTALL|re.IGNORECASE)).strip()
    )
    
    # Clean headlines if they exist
    if 'article_headline' in df.columns:
        df['article_headline'] = df['article_headline'].astype(str).apply(lambda x: emoji_pattern.sub('', x).strip())
    
    # Save cleaned version
    output_file = f"{os.path.splitext(input_file)[0]}_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved as: {output_file}")
    
    return output_file

# Clean the CSV
clean_csv(main_csv)  