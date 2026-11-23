# Delivery Time Estimation

## Project Overview
This project uses machine learning to estimate e-commerce delivery time based on order-related information. It also provides historical delivery analysis and compares multiple machine learning models.

## Features
- Predict estimated delivery time.
- Analyze historical delivery data.
- Visualize delivery time distributions and monthly trends.
- Compare four machine learning models.
- Explore late delivery statistics and important delivery factors.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Machine Learning Models
Four models were evaluated:

| Model | MAE (days) | RMSE (days) | R² Score |
|---|---:|---:|---:|
| Random Forest | 4.83 | 7.71 | 0.3156 |
| Gradient Boosting | 4.98 | 7.82 | 0.2952 |
| Linear Regression | 5.29 | 8.13 | 0.2384 |
| Decision Tree | 5.44 | 8.94 | 0.0786 |

Random Forest performed best among these models based on the lowest Mean Absolute Error (MAE).

## Project Structure
- `app.py` — Main project entry point
- `src/` — Data preprocessing, feature engineering, model training, and application code
- `models/` — Trained machine learning models
- `reports/` — Analysis charts
- `data/` — Project data files

## How to Run
1. Install Python 3 and the required dependencies.
2. Activate your virtual environment.
3. Install the dependencies listed in `requirements.txt` (if provided).
4. Run the Streamlit application using the correct app file:

   `streamlit run src/app.py`

## Evaluation
The models were compared using MAE, RMSE, and R² score. Lower MAE and RMSE indicate smaller prediction errors, while a higher R² indicates that the model explains more variation in delivery time.

## Note
Predictions are estimates, not guaranteed delivery dates. Model performance depends on the available data and selected features.
