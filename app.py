import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Reliance Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Reliance Industries Stock Price Forecasting")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/reliance_stock_data.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


df = load_data()

st.subheader("Stock Price Data")

st.dataframe(df.tail(10), use_container_width=True)

# -----------------------------
# Select price
# -----------------------------
price_column = "Close"

ts = df[price_column].dropna()

# -----------------------------
# Plot historical data
# -----------------------------
st.subheader("Historical Closing Price")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(ts.index, ts.values)

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.set_title("Reliance Industries Closing Price")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# -----------------------------
# Model selection
# -----------------------------
st.sidebar.header("Forecast Settings")

model_name = st.sidebar.selectbox(
    "Select Forecasting Model",
    [
        "ARIMA",
        "SARIMA",
        "Holt-Winters"
    ]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=90,
    value=30
)

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(model_name, values):

    series = pd.Series(values)

    if model_name == "ARIMA":

        model = ARIMA(
            series,
            order=(1, 1, 1)
        )

        fitted_model = model.fit()

    elif model_name == "SARIMA":

        model = ARIMA(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 5)
        )

        fitted_model = model.fit()

    elif model_name == "Holt-Winters":

        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        )

        fitted_model = model.fit()

    return fitted_model


# -----------------------------
# Forecast
# -----------------------------
if st.sidebar.button("Generate Forecast"):

    with st.spinner("Training model and generating forecast..."):

        try:

            model = train_model(
                model_name,
                tuple(ts.values)
            )

            forecast = model.forecast(
                steps=forecast_days
            )

            last_date = ts.index[-1]

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )

            forecast_df = pd.DataFrame(
                {
                    "Date": future_dates,
                    "Forecast": forecast.values
                }
            )

            # -----------------------------
            # Forecast table
            # -----------------------------
            st.subheader(
                f"{model_name} Forecast"
            )

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # -----------------------------
            # Forecast chart
            # -----------------------------
            st.subheader(
                "Historical + Forecast"
            )

            fig2, ax2 = plt.subplots(
                figsize=(12, 5)
            )

            ax2.plot(
                ts.index,
                ts.values,
                label="Historical"
            )

            ax2.plot(
                future_dates,
                forecast.values,
                label="Forecast"
            )

            ax2.set_xlabel("Date")
            ax2.set_ylabel("Closing Price")

            ax2.set_title(
                f"Reliance Stock Forecast - {model_name}"
            )

            ax2.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig2)

            # -----------------------------
            # Download forecast
            # -----------------------------
            csv = forecast_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name="reliance_forecast.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Forecasting error: {e}"
            )

# -----------------------------
# Summary
# -----------------------------
st.subheader("Project Summary")

st.write(
    "This application analyzes Reliance Industries "
    "historical stock prices and generates future "
    "closing-price forecasts using time-series models."
)
