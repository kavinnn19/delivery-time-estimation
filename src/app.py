import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Delivery Time Estimation",
    page_icon="📦",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.joblib")


model = load_model()

st.title("📦 Delivery Time Estimation")
st.write(
    "Predict delivery duration and analyse historical delivery data "
    "using our trained Random Forest model."
)

st.header("🚚 Predict Delivery Time")

with st.form("delivery_form"):
    st.subheader("Order Details")

    col1, col2 = st.columns(2)

    with col1:
        item_count = st.number_input(
            "Number of Items", min_value=1, value=1
        )
        total_price = st.number_input(
            "Total Price", min_value=0.0, value=100.0
        )
        total_freight = st.number_input(
            "Total Freight Cost", min_value=0.0, value=20.0
        )
        unique_sellers = st.number_input(
            "Number of Sellers", min_value=1, value=1
        )
        purchase_date = st.date_input("Purchase Date")
        purchase_time = st.time_input("Purchase Time")

    with col2:
        product_categories = st.text_input(
            "Product Category", value="beleza_saude"
        )
        avg_product_weight_g = st.number_input(
            "Average Product Weight (g)", min_value=0.0, value=1000.0
        )
        avg_product_length_cm = st.number_input(
            "Average Product Length (cm)", min_value=0.0, value=20.0
        )
        avg_product_height_cm = st.number_input(
            "Average Product Height (cm)", min_value=0.0, value=10.0
        )
        avg_product_width_cm = st.number_input(
            "Average Product Width (cm)", min_value=0.0, value=15.0
        )

    st.subheader("Seller and Customer Details")

    col3, col4 = st.columns(2)

    with col3:
        seller_cities = st.text_input(
            "Seller City", value="sao paulo"
        )
        seller_states = st.text_input(
            "Seller State", value="SP"
        )

    with col4:
        customer_city = st.text_input(
            "Customer City", value="sao paulo"
        )
        customer_state = st.text_input(
            "Customer State", value="SP"
        )

    submitted = st.form_submit_button("Predict Delivery Time")


if submitted:
    purchase_datetime = datetime.combine(
        purchase_date, purchase_time
    )

    input_data = pd.DataFrame([{
        "item_count": item_count,
        "total_price": total_price,
        "total_freight": total_freight,
        "unique_sellers": unique_sellers,
        "avg_product_weight_g": avg_product_weight_g,
        "avg_product_length_cm": avg_product_length_cm,
        "avg_product_height_cm": avg_product_height_cm,
        "avg_product_width_cm": avg_product_width_cm,
        "purchase_month": purchase_datetime.month,
        "purchase_dayofweek": purchase_datetime.weekday(),
        "purchase_hour": purchase_datetime.hour,
        "product_categories": product_categories,
        "seller_cities": seller_cities,
        "seller_states": seller_states,
        "customer_state": customer_state,
        "customer_city": customer_city
    }])

    try:
        predicted_days = max(
            0, float(model.predict(input_data)[0])
        )

        estimated_date = purchase_datetime + timedelta(
            days=predicted_days
        )

        st.success("Prediction completed!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Delivery Duration",
            f"{predicted_days:.1f} days"
        )

        col2.metric(
            "Estimated Delivery Date",
            estimated_date.strftime("%d %b %Y")
        )

        st.caption(
            "This is a model estimate, not a guaranteed delivery date."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()
st.header("📊 Historical Delivery Analysis")

uploaded_file = st.file_uploader(
    "Upload Historical Delivery CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        history = pd.read_csv(uploaded_file)

        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)

        col1.metric("Total Rows", f"{len(history):,}")
        col2.metric("Total Columns", f"{len(history.columns):,}")

        st.write("Preview of uploaded data")

        st.dataframe(
            history.head(10),
            use_container_width=True
        )

        if "delivery_time_days" in history.columns:
            delivery = pd.to_numeric(
                history["delivery_time_days"],
                errors="coerce"
            ).dropna()

            delivery = delivery[delivery >= 0]

            if not delivery.empty:
                st.subheader("Delivery Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Valid Orders",
                    f"{len(delivery):,}"
                )

                col2.metric(
                    "Average Delivery",
                    f"{delivery.mean():.2f} days"
                )

                col3.metric(
                    "Median Delivery",
                    f"{delivery.median():.2f} days"
                )

                st.subheader("Delivery Time Distribution")

                histogram = pd.cut(
                    delivery,
                    bins=10,
                    include_lowest=True
                ).value_counts().sort_index()

                histogram.index = histogram.index.astype(str)

                st.bar_chart(histogram)

            else:
                st.warning(
                    "No valid delivery_time_days values were found."
                )
        else:
            st.info(
                "This CSV has no delivery_time_days column, "
                "so delivery-duration statistics cannot be calculated."
            )

        required_columns = {
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        }

        if required_columns.issubset(history.columns):
            actual = pd.to_datetime(
                history["order_delivered_customer_date"],
                errors="coerce"
            )

            estimated = pd.to_datetime(
                history["order_estimated_delivery_date"],
                errors="coerce"
            )

            delay_days = (
                actual - estimated
            ).dt.total_seconds() / 86400

            valid_delay = delay_days.dropna()
            late_orders = valid_delay[valid_delay > 0]

            st.subheader("⏰ Late Delivery Analysis")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Orders With Both Dates",
                f"{len(valid_delay):,}"
            )

            col2.metric(
                "Late Orders",
                f"{len(late_orders):,}"
            )

            col3.metric(
                "Late Delivery Rate",
                f"{len(late_orders) / len(valid_delay) * 100:.1f}%"
                if len(valid_delay) else "N/A"
            )

            st.caption(
                "A late order is one delivered after its estimated "
                "delivery date. Only rows with both dates are counted."
            )

            history["delay_days"] = delay_days

        st.divider()
        st.header("🔎 Important Factors & Delivery Insights")

        if "delivery_time_days" in history.columns:

            factor_columns = [
                "item_count",
                "total_price",
                "total_freight",
                "unique_sellers",
                "avg_product_weight_g",
                "avg_product_length_cm",
                "avg_product_height_cm",
                "avg_product_width_cm"
            ]

            available_factors = [
                column for column in factor_columns
                if column in history.columns
            ]

            if available_factors:
                factor_data = history[available_factors].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                factor_data["delivery_time_days"] = pd.to_numeric(
                    history["delivery_time_days"],
                    errors="coerce"
                )

                correlations = (
                    factor_data.corr(numeric_only=True)
                    ["delivery_time_days"]
                    .drop("delivery_time_days")
                    .dropna()
                )

                if not correlations.empty:
                    correlations = correlations.loc[
                        correlations.abs().sort_values(
                            ascending=False
                        ).index
                    ]

                    st.subheader(
                        "Factors Associated with Delivery Duration"
                    )

                    st.caption(
                        "Correlation measures association, not proof "
                        "that a factor causes delivery delays."
                    )

                    st.bar_chart(correlations)

                    correlation_table = correlations.rename(
                        "Correlation with Delivery Time"
                    ).to_frame()

                    st.dataframe(
                        correlation_table.round(3),
                        use_container_width=True
                    )

                    strongest_factor = correlations.abs().idxmax()
                    strongest_value = correlations[strongest_factor]

                    if strongest_value > 0:
                        st.info(
                            f"📈 **{strongest_factor}** has the strongest "
                            "positive correlation among the available "
                            "factors. Higher values tend to be associated "
                            "with longer delivery times."
                        )
                    elif strongest_value < 0:
                        st.info(
                            f"📉 **{strongest_factor}** has the strongest "
                            "negative correlation among the available "
                            "factors. Higher values tend to be associated "
                            "with shorter delivery times."
                        )
                    else:
                        st.info(
                            "The available factors show no meaningful "
                            "linear correlation with delivery duration."
                        )

                    st.write(
                        "These results describe patterns in the uploaded "
                        "historical data. They do not establish causation."
                    )

                else:
                    st.info(
                        "Not enough valid numeric data to calculate "
                        "factor correlations."
                    )

            else:
                st.info(
                    "No supported factor columns were found. "
                    "Upload the model_features.csv dataset."
                )

        else:
            st.info(
                "Upload a dataset containing delivery_time_days "
                "to analyse delivery factors."
            )

        if (
            "customer_state" in history.columns
            and "delivery_time_days" in history.columns
        ):
            st.divider()
            st.header("📍 Delivery Time by Customer State")

            state_data = history[
                ["customer_state", "delivery_time_days"]
            ].copy()

            state_data["delivery_time_days"] = pd.to_numeric(
                state_data["delivery_time_days"],
                errors="coerce"
            )

            state_data = state_data.dropna(
                subset=["customer_state", "delivery_time_days"]
            )

            state_data = state_data[
                state_data["delivery_time_days"] >= 0
            ]

            if not state_data.empty:
                state_summary = (
                    state_data.groupby("customer_state")
                    ["delivery_time_days"]
                    .agg(["mean", "count"])
                    .query("count >= 10")
                    .sort_values("mean", ascending=False)
                    .head(10)
                )

                if not state_summary.empty:
                    state_summary = state_summary.rename(
                        columns={
                            "mean": "Average Delivery (Days)",
                            "count": "Number of Orders"
                        }
                    )

                    st.write(
                        "States with the highest average delivery "
                        "duration (minimum 10 orders per state)."
                    )

                    st.bar_chart(
                        state_summary["Average Delivery (Days)"]
                    )

                    st.dataframe(
                        state_summary.round(2),
                        use_container_width=True
                    )
                else:
                    st.info(
                        "Not enough orders per state to display "
                        "a reliable comparison."
                    )
            else:
                st.info("No valid state delivery records were found.")

        st.divider()
        st.subheader("📥 Download Report")

        st.download_button(
            label="⬇️ Download Analysis CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="delivery_analysis_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Could not analyse this CSV: {e}")

st.divider()
st.header("🤖 ML Model Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Linear Regression",
        "Decision Tree"
    ],
    "MAE (Days)": [4.83, 4.98, 5.29, 5.44],
    "RMSE (Days)": [7.71, 7.82, 8.13, 8.94],
    "R² Score": [0.3156, 0.2952, 0.2384, 0.0786]
})

st.write(
    "Comparison of four trained models using the existing "
    "test-set evaluation results. Lower MAE and RMSE are better; "
    "a higher R² score is better."
)

st.dataframe(
    comparison_data.style.format({
        "MAE (Days)": "{:.2f}",
        "RMSE (Days)": "{:.2f}",
        "R² Score": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

best_model_row = comparison_data.loc[
    comparison_data["MAE (Days)"].idxmin()
]

col1, col2, col3 = st.columns(3)
col1.metric("Best Model by MAE", best_model_row["Model"])
col2.metric("Lowest MAE", f'{best_model_row["MAE (Days)"]:.2f} days')
col3.metric("Best Model R²", f'{best_model_row["R² Score"]:.4f}')

st.subheader("MAE Comparison")
st.caption("Lower MAE means lower average absolute prediction error.")
st.bar_chart(comparison_data.set_index("Model")[["MAE (Days)"]])

st.subheader("RMSE Comparison")
st.caption("RMSE penalizes larger prediction errors more strongly.")
st.bar_chart(comparison_data.set_index("Model")[["RMSE (Days)"]])

st.subheader("R² Score Comparison")
st.caption(
    "R² indicates how much variation the model explains on the evaluation "
    "dataset. It can be negative for a model that performs worse than a "
    "mean-value baseline."
)
st.bar_chart(comparison_data.set_index("Model")[["R² Score"]])

st.info(
    "Random Forest performed best among these four models based on the "
    "lowest MAE in the recorded evaluation results. These are saved "
    "evaluation results; this dashboard section does not retrain models."
)
