import pandas as pd
import numpy as np
from scipy import stats

# Load data
llm_df = pd.read_csv("weekly_predictions_inflation_gemini-2_5-pro_with_reflection_14 day.csv")
ablation_df = pd.read_csv("weekly_predictions_inflation_gemini-2_5-pro_with_reflection_ABLATION.csv")

# Convert dates to match
llm_df['prediction_date'] = pd.to_datetime(llm_df['prediction_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
llm_df['target_date'] = pd.to_datetime(llm_df['target_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

ablation_df['prediction_date'] = pd.to_datetime(ablation_df['prediction_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
ablation_df['target_date'] = pd.to_datetime(ablation_df['target_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# Merge datasets
merged_df = llm_df.merge(ablation_df[['prediction_date', 'target_date', 'prediction', 'initial_prediction']], 
                        on=['prediction_date', 'target_date'], 
                        suffixes=('_llm', '_ablation'))

complete_df = merged_df.dropna(subset=['actual_nowcast'])

# Calculate errors
llm_initial_error = abs(complete_df['initial_prediction_llm'] - complete_df['actual_nowcast'])
ablation_initial_error = abs(complete_df['initial_prediction_ablation'] - complete_df['actual_nowcast'])
llm_reflection_error = abs(complete_df['prediction_llm'] - complete_df['actual_nowcast'])
ablation_reflection_error = abs(complete_df['prediction_ablation'] - complete_df['actual_nowcast'])

# Statistics
llm_initial_mae = llm_initial_error.mean()
llm_initial_rmse = np.sqrt((llm_initial_error**2).mean())
ablation_initial_mae = ablation_initial_error.mean()
ablation_initial_rmse = np.sqrt((ablation_initial_error**2).mean())
llm_reflection_mae = llm_reflection_error.mean()
llm_reflection_rmse = np.sqrt((llm_reflection_error**2).mean())
ablation_reflection_mae = ablation_reflection_error.mean()
ablation_reflection_rmse = np.sqrt((ablation_reflection_error**2).mean())

# Regret calculations
llm_initial_regret = (llm_initial_error > ablation_initial_error).sum()
ablation_initial_regret = (ablation_initial_error > llm_initial_error).sum()
llm_reflection_regret = (llm_reflection_error > ablation_reflection_error).sum()
ablation_reflection_regret = (ablation_reflection_error > llm_reflection_error).sum()

initial_ties = (llm_initial_error == ablation_initial_error).sum()
reflection_ties = (llm_reflection_error == ablation_reflection_error).sum()

# Print results
print("Initial Predictions Comparison:")
print(f"  LLM Initial MAE: {llm_initial_mae:.4f}")
print(f"  LLM Initial RMSE: {llm_initial_rmse:.4f}")
print(f"  LLM Initial regret: {llm_initial_regret} ({llm_initial_regret/len(complete_df)*100:.1f}%)")

print(f"\n  Ablation Initial MAE: {ablation_initial_mae:.4f}")
print(f"  Ablation Initial RMSE: {ablation_initial_rmse:.4f}")
print(f"  Ablation Initial regret: {ablation_initial_regret} ({ablation_initial_regret/len(complete_df)*100:.1f}%)")

print(f"\n  Difference (LLM - Ablation): {llm_initial_mae - ablation_initial_mae:.4f}")
print(f"  LLM better than Ablation: {llm_initial_mae < ablation_initial_mae}")
print(f"  Ties: {initial_ties}")

print("\nReflection Predictions Comparison:")
print(f"  LLM Reflection MAE: {llm_reflection_mae:.4f}")
print(f"  LLM Reflection RMSE: {llm_reflection_rmse:.4f}")
print(f"  LLM Reflection regret: {llm_reflection_regret} ({llm_reflection_regret/len(complete_df)*100:.1f}%)")

print(f"\n  Ablation Reflection MAE: {ablation_reflection_mae:.4f}")
print(f"  Ablation Reflection RMSE: {ablation_reflection_rmse:.4f}")
print(f"  Ablation Reflection regret: {ablation_reflection_regret} ({ablation_reflection_regret/len(complete_df)*100:.1f}%)")

print(f"\n  Difference (LLM - Ablation): {llm_reflection_mae - ablation_reflection_mae:.4f}")
print(f"  LLM better than Ablation: {llm_reflection_mae < ablation_reflection_mae}")
print(f"  Ties: {reflection_ties}")

print(f"\nMatched predictions: {len(complete_df)}")

# Statistical tests
t_stat_initial, p_value_initial = stats.ttest_rel(llm_initial_error, ablation_initial_error)
t_stat_reflection, p_value_reflection = stats.ttest_rel(llm_reflection_error, ablation_reflection_error)

print(f"\nStatistical Significance:")
print(f"  Initial predictions p-value: {p_value_initial:.4f} (significant: {p_value_initial < 0.05})")
print(f"  Reflection predictions p-value: {p_value_reflection:.4f} (significant: {p_value_reflection < 0.05})")

# Save results - CSV 1: 4 lines
results = {
    'model': ['LLM_initial', 'Ablation_initial', 'LLM_reflection', 'Ablation_reflection'],
    'mae': [llm_initial_mae, ablation_initial_mae, llm_reflection_mae, ablation_reflection_mae],
    'rmse': [llm_initial_rmse, ablation_initial_rmse, llm_reflection_rmse, ablation_reflection_rmse],
    'regret_count': [llm_initial_regret, ablation_initial_regret, llm_reflection_regret, ablation_reflection_regret],
    'regret_percentage': [llm_initial_regret/len(complete_df)*100, ablation_initial_regret/len(complete_df)*100,
                         llm_reflection_regret/len(complete_df)*100, ablation_reflection_regret/len(complete_df)*100],
    'matched_predictions': [len(complete_df), len(complete_df), len(complete_df), len(complete_df)]
}

# CSV 2: 2 lines of comparisons
comparison_stats = [
    {
        'comparison': 'LLM_vs_Ablation_initial',
        'mae_difference': llm_initial_mae - ablation_initial_mae,
        'rmse_difference': llm_initial_rmse - ablation_initial_rmse,
        't_statistic': t_stat_initial,
        'p_value': p_value_initial,
        'significant_5pct': p_value_initial < 0.05,
        'ties': initial_ties
    },
    {
        'comparison': 'LLM_vs_Ablation_reflection',
        'mae_difference': llm_reflection_mae - ablation_reflection_mae,
        'rmse_difference': llm_reflection_rmse - ablation_reflection_rmse,
        'llm_better_mae': llm_reflection_mae < ablation_reflection_mae,
        'llm_better_rmse': llm_reflection_rmse < ablation_reflection_rmse,
        't_statistic': t_stat_reflection,
        'p_value': p_value_reflection,
        'significant_5pct': p_value_reflection < 0.05,
        'ties': reflection_ties
    }
]

results_df = pd.DataFrame(results)
results_df.to_csv("LLM vs Ablation 14 days.csv", index=False)

comparison_df = pd.DataFrame(comparison_stats)
comparison_df.to_csv("LLM vs Ablation 14 days_comparison.csv", index=False)

print(f"\nResults saved to 'LLM vs Ablation 14 days.csv' (4 lines)")
print(f"Comparison saved to 'LLM vs Ablation 14 days_comparison.csv' (2 lines)")