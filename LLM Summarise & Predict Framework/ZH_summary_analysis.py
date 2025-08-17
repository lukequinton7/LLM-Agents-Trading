import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer


#save type
"""
type = "inflation"  
"""

type = "rates"  # for save

# Load dataframes

#silver standard
"""
silver_df = pd.read_csv('ZH_article_summaries_inflation_gemini-2_5-pro.csv')
"""
silver_df = pd.read_csv('ZH_article_summaries_rates_gemini-2_5-pro.csv')


#rates

#inflation models
"""
models = {
    'Gemma 1B': pd.read_csv('ZH_article_summaries_inflation_gemma3_1b.csv'),
    'Gemma 4B': pd.read_csv('ZH_article_summaries_inflation_gemma3_4b.csv'),
    'Gemma 12B': pd.read_csv('ZH_article_summaries_inflation_gemma3_12b.csv'),
    'Llama 3.1 8B': pd.read_csv('ZH_article_summaries_inflation_llama3_1_8b.csv')
}
"""

#rates models
models = {
    'Gemma 1B': pd.read_csv('ZH_article_summaries_rates_gemma3_1b.csv'),
    'Gemma 4B': pd.read_csv('ZH_article_summaries_rates_gemma3_4b.csv'),
    'Gemma 12B': pd.read_csv('ZH_article_summaries_rates_gemma3_12b.csv'),
    'Llama 3.1 8B': pd.read_csv('ZH_article_summaries_rates_llama3_1_8b.csv')
}



def analyse_model(silver_df, model_df):
    """Analyse a model against silver standard"""
    merged = pd.merge(silver_df, model_df, on=['datetime', 'headline'], suffixes=('_silver', '_model'))
    
    # Check if the summary CONTAINS the key phrase, ignoring case.
    # ~ inverts the boolean result,  y_true is True if a summary exists.
    y_true = ~merged['summary_silver'].str.contains('no relevant content', case=False, na=False)
    y_pred = ~merged['summary_model'].str.contains('no relevant content', case=False, na=False)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    n = len(y_true)
    se = np.sqrt((acc * (1 - acc)) / n)
    ci_low = acc - 1.96 * se
    ci_high = acc + 1.96 * se
    
    # word count analysis
    word_counts = merged['summary_model'].str.split().str.len()
    pct_over_150 = (word_counts > 150).mean() * 100
    pct_over_300 = (word_counts > 300).mean() * 100
    
    # calculating the percentage.
    no_relevant_pct = merged['summary_model'].str.contains('no relevant content', case=False, na=False).mean() * 100
    
    # ROUGE score calculation
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores_relevant = []
    

    for idx, row in merged.iterrows():
        reference = row['summary_silver']
        hypothesis = row['summary_model']
        
        # calc roughe scores for relevant content
        if 'no relevant content' not in reference.lower():
            scores = scorer.score(reference, hypothesis)
            rouge_scores_relevant.append(scores)
    
    if rouge_scores_relevant:
        avg_rouge1_rel = np.mean([score['rouge1'].fmeasure for score in rouge_scores_relevant])
        avg_rouge2_rel = np.mean([score['rouge2'].fmeasure for score in rouge_scores_relevant])
        avg_rougeL_rel = np.mean([score['rougeL'].fmeasure for score in rouge_scores_relevant])
    else:
        avg_rouge1_rel, avg_rouge2_rel, avg_rougeL_rel = 0.0, 0.0, 0.0
    
    return [acc, ci_low, ci_high, prec, rec, f1, pct_over_150, pct_over_300, no_relevant_pct, avg_rouge1_rel, avg_rouge2_rel, avg_rougeL_rel]


# Calculate silver standard stats- ie gemini 
silver_word_counts = silver_df['summary'].str.split().str.len()
silver_stats = {
    'over_150': (silver_word_counts > 150).mean() * 100,
    'over_300': (silver_word_counts > 300).mean() * 100,
    'no_rel': silver_df['summary'].str.contains('no relevant content', case=False, na=False).mean() * 100
}
# Analyse all models
results = {}
for name, df in models.items():
    results[name] = analyse_model(silver_df, df)

# Calculate silver standard stats- high level results
silver_word_counts = silver_df['summary'].str.split().str.len()
silver_stats = {
    'over_150': (silver_word_counts > 150).mean() * 100,
    'over_300': (silver_word_counts > 300).mean() * 100,
    'no_rel': (silver_df['summary'] == 'No relevant content').mean() * 100
}

# Create Table 1- Performance Metrics
performance_data = {
    'Model': list(results.keys()),
    'Accuracy': [f"{results[name][0]:.1%}" for name in results.keys()],
    '95% CI': [f"({results[name][1]:.1%} - {results[name][2]:.1%})" for name in results.keys()],
    'Precision': [f"{results[name][3]:.2f}" for name in results.keys()],
    'Recall': [f"{results[name][4]:.2f}" for name in results.keys()],
    'F1-Score': [f"{results[name][5]:.2f}" for name in results.keys()],
    'ROUGE-1': [f"{results[name][9]:.3f}" for name in results.keys()],
    'ROUGE-2': [f"{results[name][10]:.3f}" for name in results.keys()],
    'ROUGE-L': [f"{results[name][11]:.3f}" for name in results.keys()]
}

performance_df = pd.DataFrame(performance_data)

# Create Table 2: Output Characteristics
output_data = {
    'Model': ['Silver Standard'] + list(results.keys()),
    '"No Relevant Content" %': [f"{silver_stats['no_rel']:.1f}%"] + [f"{results[name][8]:.1f}%" for name in results.keys()],
    '>150 words %': [f"{silver_stats['over_150']:.1f}%"] + [f"{results[name][6]:.1f}%" for name in results.keys()],
    '>300 words %': [f"{silver_stats['over_300']:.1f}%"] + [f"{results[name][7]:.1f}%" for name in results.keys()]
}

output_df = pd.DataFrame(output_data)

# Print both tables
print("Table 1: Performance Metrics")
print("=" * 50)
print(performance_df.to_string(index=False))
print("\n")

print("Table 2: Output Characteristics")
print("=" * 50)
print(output_df.to_string(index=False))

#save output df to csv
output_filename = f"ZH_summary_analysis_{type}.csv"
output_df.to_csv(output_filename, index=False)

#save performance df to csv
performance_filename = f"ZH_performance_metrics_{type}.csv"
performance_df.to_csv(performance_filename, index=False)


