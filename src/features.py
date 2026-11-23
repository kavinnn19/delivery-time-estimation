import pandas as pd

items = pd.read_csv("data/olist_order_items_dataset.csv")
customers = pd.read_csv("data/olist_customers_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")
orders = pd.read_csv("data/processed_orders.csv")

items = items.merge(
    products[
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ],
    on="product_id",
    how="left",
)

items = items.merge(
    sellers[["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left",
)

order_features = items.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    unique_sellers=("seller_id", "nunique"),
    product_categories=("product_category_name",
                        lambda x: ", ".join(sorted(x.dropna().unique()))),
    avg_product_weight_g=("product_weight_g", "mean"),
    avg_product_length_cm=("product_length_cm", "mean"),
    avg_product_height_cm=("product_height_cm", "mean"),
    avg_product_width_cm=("product_width_cm", "mean"),
    seller_cities=("seller_city",
                   lambda x: ", ".join(sorted(x.dropna().unique()))),
    seller_states=("seller_state",
                   lambda x: ", ".join(sorted(x.dropna().unique()))),
).reset_index()

features = orders.merge(order_features, on="order_id", how="left")

customer_locations = customers[
    ["customer_id", "customer_city", "customer_state"]
].drop_duplicates(subset=["customer_id"])

features = features.merge(
    customer_locations,
    on="customer_id",
    how="left",
)

features.to_csv("data/model_features.csv", index=False)

print("Combined features created successfully!")
print("Total records:", len(features))

print("\nSelected columns:")
print(
    features[
        [
            "order_id",
            "item_count",
            "product_categories",
            "avg_product_weight_g",
            "seller_cities",
            "seller_states",
            "customer_city",
            "customer_state",
            "delivery_time_days",
        ]
    ].head()
)

print("\nMissing values in selected features:")
print(
    features[
        [
            "item_count",
            "product_categories",
            "avg_product_weight_g",
            "seller_cities",
            "seller_states",
            "customer_city",
            "customer_state",
        ]
    ].isnull().sum()
)
