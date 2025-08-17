
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# FinBERT is a CLASSIFICATION model. It is pre-trained to only output
# one of three labels: 'positive', 'negative', or 'neutral'.

# Define file paths and load the Finbert model 

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("Model and tokenizer loaded successfully.")


csv_input_file = "test tweets/+1 sample.csv"  # likely inflation sample- ie. seeking +1 outcome

# Read and Process the CSV data ---
try:

    #load tweets from csv to df
    df = pd.read_csv(csv_input_file)

    if 'text' in df.columns:
        # We extract ONLY the text data for FinBERT ie. no question or prompt

        #df to list
        tweets_data = " ".join(df['text'].astype(str).tolist())
        print(f"loaded and combined {len(df)} tweets.")
    else:
        tweets_data = ""
        print("Error: 'text' column not found in the CSV file.")

except FileNotFoundError:
    print(f"Error: The file '{csv_input_file}' was not found.")
    tweets_data = ""
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    tweets_data = ""


# analyse Sentiment and Map the Result 

if tweets_data:
    print("\n Analyzing Sentiment (Finbert)")

    # Tokenize the tweet data. This prepares the text for the model.
    inputs = tokenizer(
        tweets_data,
        return_tensors="pt",
        padding=True,
        truncation=True, # IMPORTANT: Truncates long inputs to the model's 512 token limit
        max_length=512
    )

    # get classification output
    with torch.no_grad():
        outputs = model(**inputs)

    # get raw prediction (logits) and find the highest scoring class
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    sentiment_label = model.config.id2label[predicted_class_id] # This will be 'positive', 'negative', or 'neutral'
    
    print(f"FinBERT Raw Output (Classification Label): '{sentiment_label}'")

    # map step
    #FinBERT is simple classifier (pos, neutral, neg)-map for desired outcomes

    sentiment_score_map = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    final_score = sentiment_score_map.get(sentiment_label, 0)

    # Numerical mapped output
    print(f"\nSentiment Score: {final_score}")

else:
    print("\nNo tweet data was loaded. Analysis could not be performed.")
