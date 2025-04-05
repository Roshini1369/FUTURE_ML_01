import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set App Title
st.title("📈 Sales Forecasting Web App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)
    data["date"] = pd.to_datetime(data["date"])
    st.write("### Sample Data", data.head())

    # Prophet Forecasting Model
    df = data.rename(columns={"date": "ds", "sales": "y"})
    
    # Splitting Data for Accuracy Check
    train_size = int(len(df) * 0.8)  # 80% training, 20% testing
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    model = Prophet()
    model.fit(train_df)

    # Predict on Test Set
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    # Extract only forecasted values matching the test period
    test_forecast = forecast.iloc[-len(test_df):]["yhat"].values
    actual_sales = test_df["y"].values

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(actual_sales, test_forecast)
    rmse = np.sqrt(mean_squared_error(actual_sales, test_forecast))
    r2 = r2_score(actual_sales, test_forecast)

    # Display Accuracy Results
    st.write("### 📊 Model Accuracy Metrics")
    st.write(f"✔ Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"✔ Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"✔ R² Score: {r2:.2f} (Higher is better)")

    # Plot Actual vs Predicted Sales
    st.write("### 🔍 Actual vs Predicted Sales")
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["ds"], actual_sales, label="Actual Sales", color="blue")
    plt.plot(test_df["ds"], test_forecast, label="Predicted Sales", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    st.pyplot(plt)

    # Show Forecast Graph
    st.write("### 📈 Sales Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Show Trend & Seasonality
    st.write("### 📊 Trend & Seasonality Analysis")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
