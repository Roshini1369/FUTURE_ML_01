import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set App Title
st.title("📈 Sales Forecasting Web App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file:
    # Handle encoding issues
    try:
        data = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    # Date Conversion
    try:
        data["date"] = pd.to_datetime(data["date"])
    except Exception as e:
        st.error(f"Failed to parse 'date' column. Ensure it exists and is properly formatted. Error: {e}")
        st.stop()

    st.write("### Sample Data", data.head())

    # Prophet Forecasting Model
    df = data.rename(columns={"date": "ds", "sales": "y"})

    # Split Data: 80% Train, 20% Test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    model = Prophet()
    model.fit(train_df)

    # Make Future DataFrame for Prediction
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)

    # Get Forecasted vs Actual
    test_forecast = forecast.iloc[-len(test_df):]["yhat"].values
    actual_sales = test_df["y"].values

    # Metrics
    mae = mean_absolute_error(actual_sales, test_forecast)
    rmse = np.sqrt(mean_squared_error(actual_sales, test_forecast))
    r2 = r2_score(actual_sales, test_forecast)

    # Accuracy Results
    st.write("### 📊 Model Accuracy Metrics")
    st.write(f"✔ Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"✔ Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"✔ R² Score: {r2:.2f} (Higher is better)")

    # Plot Actual vs Predicted
    st.write("### 🔍 Actual vs Predicted Sales")
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["ds"], actual_sales, label="Actual Sales", color="blue")
    plt.plot(test_df["ds"], test_forecast, label="Predicted Sales", linestyle="dashed", color="red")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    st.pyplot(plt)

    # Forecast Plot
    st.write("### 📈 Sales Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Trend & Seasonality
    st.write("### 📊 Trend & Seasonality Analysis")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
