import pandas as pd
import matplotlib.pyplot as plt
# manually enter your results
results = pd.DataFrame({
    "Model": ["ARIMA", "Hybrid (LSTM + HMM)"],
    "RMSE": [0.007908, 0.010935],
    "Direction Accuracy": [0.5333, 0.5009]
})

print("\n=== FINAL COMPARISON ===")
print(results)

results.to_csv("results/final_comparison.csv", index=False)


plt.figure()
plt.bar(results["Model"], results["RMSE"])
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.savefig("results/rmse_comparison.png")
plt.show()