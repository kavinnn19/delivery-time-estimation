
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load combined dataset
df = pd.read_csv("data/model_features.csv")

# Create features from order purchase time
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"], errors="coerce"
)
df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

# Use only information available when making a prediction
numeric_features = [
    "item_count",
    "total_price",
    "total_freight",
    "unique_sellers",
    "avg_product_weight_g",
    "avg_product_length_cm",
    "avg_product_height_cm",
    "avg_product_width_cm",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
]

categorical_features = [
    "product_categories",
    "seller_cities",
    "seller_states",
    "customer_state",
    "customer_city",
]

features = numeric_features + categorical_features
target = "delivery_time_days"

# Remove records without a valid target
df = df.dropna(subset=[target, "order_purchase_timestamp"]).copy()

X = df[features]
y = df[target]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fill missing numeric values and encode categories
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=20,
                    ),
                ),
            ]),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

print("Training Linear Regression model with product and seller features...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results")
print("-------------------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R2 Score: {r2:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_regression_model.joblib")

print("\nModel saved to models/linear_regression_model.joblib")
