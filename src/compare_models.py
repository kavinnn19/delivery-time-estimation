
import os
import pandas as pd

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
    ],
    "MAE_days": [5.29, 4.83, 5.44, 4.98],
    "RMSE_days": [8.13, 7.71, 8.94, 7.82],
    "R2_Score": [0.2384, 0.3156, 0.0786, 0.2952],
})

results = results.sort_values("MAE_days").reset_index(drop=True)

print("\nDelivery Time Estimation - Four Model Comparison")
print("=" * 55)
print(results.to_string(index=False))

best_model = results.iloc[0]["Model"]
print(f"\nBest model based on MAE: {best_model}")

os.makedirs("reports", exist_ok=True)
results.to_csv("reports/model_comparison.csv", index=False)

print("\nComparison report saved to reports/model_comparison.csv")
