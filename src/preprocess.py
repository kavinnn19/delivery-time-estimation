
import pandas as pd
from pathlib import Path

data_path = Path("data/olist_orders_dataset.csv")
df = pd.read_csv(data_path)

df = df[df["order_status"] == "delivered"].copy()

df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)
df["order_delivered_customer_date"] = pd.to_datetime(
    df["order_delivered_customer_date"]
)

df = df.dropna(subset=["order_delivered_customer_date"])

df["delivery_time_days"] = (
    df["order_delivered_customer_date"]
    - df["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)

df = df[df["delivery_time_days"] >= 0].copy()

output_path = Path("data/processed_orders.csv")
df.to_csv(output_path, index=False)

print("Preprocessing completed successfully!")
print("Total usable orders:", len(df))
print("Average delivery time:", round(df["delivery_time_days"].mean(), 2), "days")
print("Saved file:", output_path)
