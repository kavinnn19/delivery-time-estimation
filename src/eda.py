
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed_orders.csv")

Path("reports").mkdir(exist_ok=True)

plt.figure(figsize=(10, 6))
plt.hist(df["delivery_time_days"], bins=50, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("reports/delivery_time_distribution.png")
plt.close()

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)
df["purchase_month"] = (
    df["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

monthly_delivery = df.groupby("purchase_month")[
    "delivery_time_days"
].mean()

plt.figure(figsize=(12, 6))
monthly_delivery.plot()
plt.title("Average Delivery Time by Month")
plt.xlabel("Purchase Month")
plt.ylabel("Average Delivery Time (Days)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/monthly_average_delivery_time.png")
plt.close()

print("EDA completed successfully!")
print("Graph 1: reports/delivery_time_distribution.png")
print("Graph 2: reports/monthly_average_delivery_time.png")
print("Average delivery time:", round(df["delivery_time_days"].mean(), 2), "days")
print("Median delivery time:", round(df["delivery_time_days"].median(), 2), "days")
