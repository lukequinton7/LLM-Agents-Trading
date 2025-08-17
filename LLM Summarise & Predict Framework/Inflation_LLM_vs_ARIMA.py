import pandas as pd
import numpy as np
from scipy import stats

# Load data
llm_df = pd.read_csv("weekly_predictions_inflation_gemini-2_5-pro_with_reflection_28 day.csv")
arima_df = pd.read_csv("arima_predictions_inflation_28day_benchmark.csv")

# Merge on prediction_date and target_date to align predictions
merged_df = llm_df.merge(arima_df[['prediction_date', 'target_date', 'prediction']], 
                        on=['prediction_date', 'target_date'], 
                        suffixes=('_llm', '_arima'))

# Remove incomplete rows
complete_df = merged_df.dropna(subset=['actual_nowcast'])

# Calculate errors
llm_error = abs(complete_df['prediction_llm'] - complete_df['actual_nowcast'])
llm_initial_error = abs(complete_df['initial_prediction'] - complete_df['actual_nowcast'])
arima_error = abs(complete_df['prediction_arima'] - complete_df['actual_nowcast'])

# LLM Statistics
llm_mae = llm_error.mean()
llm_rmse = np.sqrt((llm_error**2).mean())
llm_regret = (llm_error > arima_error).sum()  # How often LLM worse than ARIMA
llm_regret_pct = (llm_regret / len(complete_df)) * 100

# ARIMA Statistics
arima_mae = arima_error.mean()
arima_rmse = np.sqrt((arima_error**2).mean())
arima_regret = (arima_error > llm_error).sum()  # How often ARIMA worse than LLM
arima_regret_pct = (arima_regret / len(complete_df)) * 100

# Check for ties
ties = (llm_error == arima_error).sum()
ties_pct = (ties / len(complete_df)) * 100

# Print results
print("LLM Statistics:")
print(f"  MAE: {llm_mae:.4f}")
print(f"  RMSE: {llm_rmse:.4f}")
print(f"  Regret count: {llm_regret}")
print(f"  Regret percentage: {llm_regret_pct:.1f}%")

print("\nARIMA Statistics:")
print(f"  MAE: {arima_mae:.4f}")
print(f"  RMSE: {arima_rmse:.4f}")
print(f"  Regret count: {arima_regret}")
print(f"  Regret percentage: {arima_regret_pct:.1f}%")

print(f"\nMatched predictions: {len(complete_df)}")
print(f"Ties: {ties} ({ties_pct:.1f}%)")
print(f"Check: {llm_regret} + {arima_regret} + {ties} = {llm_regret + arima_regret + ties}")

print("\nComparison:")
print(f"  LLM MAE - ARIMA MAE: {llm_mae - arima_mae:.4f}")
print(f"  LLM RMSE - ARIMA RMSE: {llm_rmse - arima_rmse:.4f}")
print(f"  LLM better than ARIMA (MAE): {llm_mae < arima_mae}")
print(f"  LLM better than ARIMA (RMSE): {llm_rmse < arima_rmse}")

# Statistical significance test
t_stat, p_value = stats.ttest_rel(llm_error, arima_error)

print(f"\nStatistical Significance:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at 5%: {p_value < 0.05}")

# Save results to CSV
results = {
    'model': ['LLM', 'ARIMA'],
    'mae': [llm_mae, arima_mae],
    'rmse': [llm_rmse, arima_rmse],
    'regret_count': [llm_regret, arima_regret],
    'regret_percentage': [llm_regret_pct, arima_regret_pct],
    'matched_predictions': [len(complete_df), len(complete_df)]
}

comparison_stats = {
    'mae_difference': llm_mae - arima_mae,
    'rmse_difference': llm_rmse - arima_rmse,
    'llm_better_mae': llm_mae < arima_mae,
    'llm_better_rmse': llm_rmse < arima_rmse,
    't_statistic': t_stat,
    'p_value': p_value,
    'significant_5pct': p_value < 0.05
}

# Save model results
results_df = pd.DataFrame(results)
results_df.to_csv("LLM vs ARIMA 28 days.csv", index=False)

# Save comparison stats
comparison_df = pd.DataFrame([comparison_stats])
comparison_df.to_csv("LLM vs ARIMA 28 days_comparison.csv", index=False)

print(f"\nResults saved to 'LLM vs ARIMA 28 days.csv'")
print(f"Comparison stats saved to 'LLM vs ARIMA 28 days_comparison.csv'")