import pandas as pd
import matplotlib.pyplot as plt

# load predictions
df = pd.read_csv("results/hybrid_predictions.csv")

# plot
plt.figure(figsize=(12,6))
plt.plot(df["actual"][:200], label="Actual")
plt.plot(df["predicted"][:200], label="Predicted")

plt.title("Hybrid Model: Actual vs Predicted")
plt.legend()
plt.grid()

plt.savefig("results/hybrid_plot.png")
plt.show()