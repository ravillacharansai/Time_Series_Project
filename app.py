import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reliance Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    font-size: 14px;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
}

h1 {
    font-weight: 800;
}

h2 {
    font-weight: 700;
}

h3 {
    font-weight: 650;
}

div.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.title("📈 Reliance Industries Stock Intelligence")

st.markdown(
    """
    **Advanced Time Series Forecasting Dashboard**

    Analyze historical Reliance Industries stock prices,
    explore technical indicators, and generate future price
    forecasts using statistical time-series models.
    """
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/reliance_stock_data.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Close"]
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown(
    "### 📅 Date Range"
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date = st.sidebar.date_input(
    "Start Date",
    min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:

    st.sidebar.error(
        "Start date must be before end date."
    )

    st.stop()


filtered_df = df[
    (df["Date"].dt.date >= start_date)
    &
    (df["Date"].dt.date <= end_date)
].copy()


# =========================================================
# DATA PREPARATION
# =========================================================

filtered_df["Daily Return"] = (
    filtered_df["Close"]
    .pct_change()
    * 100
)

filtered_df["MA20"] = (
    filtered_df["Close"]
    .rolling(20)
    .mean()
)

filtered_df["MA50"] = (
    filtered_df["Close"]
    .rolling(50)
    .mean()
)


# =========================================================
# KPI SECTION
# =========================================================

latest_price = filtered_df["Close"].iloc[-1]

first_price = filtered_df["Close"].iloc[0]

price_change = (
    latest_price - first_price
)

percentage_change = (
    price_change
    / first_price
    * 100
)

highest_price = (
    filtered_df["High"].max()
)

lowest_price = (
    filtered_df["Low"].min()
)

average_volume = (
    filtered_df["Close"].count()
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Latest Close",
        f"₹{latest_price:,.2f}"
    )

with col2:

    st.metric(
        "Period Change",
        f"₹{price_change:,.2f}",
        f"{percentage_change:.2f}%"
    )

with col3:

    st.metric(
        "Period High",
        f"₹{highest_price:,.2f}"
    )

with col4:

    st.metric(
        "Period Low",
        f"₹{lowest_price:,.2f}"
    )


st.divider()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Technical Analysis",
        "🔮 Forecasting",
        "📋 Data Explorer"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.subheader(
        "Historical Closing Price"
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["Close"],
        linewidth=2
    )

    ax.set_title(
        "Reliance Industries Closing Price"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price (₹)"
    )

    ax.grid(
        alpha=0.25
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Market Statistics"
    )

    stat1, stat2, stat3 = st.columns(3)

    with stat1:

        st.metric(
            "Average Close",
            f"₹{filtered_df['Close'].mean():,.2f}"
        )

    with stat2:

        st.metric(
            "Median Close",
            f"₹{filtered_df['Close'].median():,.2f}"
        )

    with stat3:

        st.metric(
            "Volatility",
            f"{filtered_df['Daily Return'].std():.2f}%"
        )


# =========================================================
# TAB 2 — TECHNICAL ANALYSIS
# =========================================================

with tab2:

    st.subheader(
        "Moving Average Analysis"
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["Close"],
        label="Close",
        linewidth=2
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["MA20"],
        label="20-Day MA",
        linewidth=1.5
    )

    ax.plot(
        filtered_df["Date"],
        filtered_df["MA50"],
        label="50-Day MA",
        linewidth=1.5
    )

    ax.set_title(
        "Price vs Moving Averages"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price (₹)"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Daily Returns"
    )

    fig2, ax2 = plt.subplots(
        figsize=(14, 4)
    )

    ax2.plot(
        filtered_df["Date"],
        filtered_df["Daily Return"],
        linewidth=1
    )

    ax2.axhline(
        0,
        linewidth=1
    )

    ax2.set_title(
        "Daily Percentage Returns"
    )

    ax2.set_xlabel(
        "Date"
    )

    ax2.set_ylabel(
        "Return (%)"
    )

    ax2.grid(
        alpha=0.25
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig2)


# =========================================================
# TAB 3 — FORECASTING
# =========================================================

with tab3:

    st.subheader(
        "🔮 Time Series Forecasting"
    )

    st.info(
        "Select a statistical forecasting model "
        "and generate future closing-price estimates."
    )

    model_name = st.selectbox(
        "Select Forecasting Model",
        [
            "ARIMA",
            "SARIMA",
            "Holt-Winters"
        ]
    )

    forecast_days = st.slider(
        "Forecast Horizon",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )

    st.write(
        f"**Model:** {model_name}  |  "
        f"**Forecast Horizon:** {forecast_days} days"
    )

    if st.button(
        "🚀 Generate Forecast",
        type="primary"
    ):

        with st.spinner(
            "Training model and generating forecast..."
        ):

            try:

                ts = (
                    filtered_df
                    .set_index("Date")["Close"]
                    .dropna()
                )

                # -------------------------------------
                # ARIMA
                # -------------------------------------

                if model_name == "ARIMA":

                    model = ARIMA(
                        ts,
                        order=(1, 1, 1)
                    )

                    fitted_model = model.fit()

                # -------------------------------------
                # SARIMA
                # -------------------------------------

                elif model_name == "SARIMA":

                    model = ARIMA(
                        ts,
                        order=(1, 1, 1),
                        seasonal_order=(
                            1,
                            1,
                            1,
                            5
                        )
                    )

                    fitted_model = model.fit()

                # -------------------------------------
                # HOLT-WINTERS
                # -------------------------------------

                else:

                    model = ExponentialSmoothing(
                        ts,
                        trend="add",
                        seasonal=None,
                        initialization_method="estimated"
                    )

                    fitted_model = model.fit()

                # -------------------------------------
                # FORECAST
                # -------------------------------------

                forecast = fitted_model.forecast(
                    steps=forecast_days
                )

                future_dates = pd.date_range(
                    start=ts.index[-1]
                    + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq="D"
                )

                forecast_df = pd.DataFrame(
                    {
                        "Date": future_dates,
                        "Forecast": forecast.values
                    }
                )

                # -------------------------------------
                # FORECAST KPI
                # -------------------------------------

                final_forecast = (
                    forecast_df["Forecast"].iloc[-1]
                )

                forecast_change = (
                    final_forecast
                    - latest_price
                )

                forecast_percentage = (
                    forecast_change
                    / latest_price
                    * 100
                )

                f1, f2, f3 = st.columns(3)

                with f1:

                    st.metric(
                        "Last Historical Price",
                        f"₹{latest_price:,.2f}"
                    )

                with f2:

                    st.metric(
                        "Final Forecast",
                        f"₹{final_forecast:,.2f}"
                    )

                with f3:

                    st.metric(
                        "Forecast Change",
                        f"{forecast_percentage:.2f}%"
                    )

                # -------------------------------------
                # FORECAST CHART
                # -------------------------------------

                st.subheader(
                    "Historical vs Forecast"
                )

                fig3, ax3 = plt.subplots(
                    figsize=(14, 6)
                )

                ax3.plot(
                    ts.index,
                    ts.values,
                    label="Historical",
                    linewidth=2
                )

                ax3.plot(
                    future_dates,
                    forecast.values,
                    label="Forecast",
                    linewidth=2
                )

                ax3.axvline(
                    ts.index[-1],
                    linestyle="--",
                    linewidth=1
                )

                ax3.set_title(
                    f"Reliance Stock Forecast — {model_name}"
                )

                ax3.set_xlabel(
                    "Date"
                )

                ax3.set_ylabel(
                    "Price (₹)"
                )

                ax3.legend()

                ax3.grid(
                    alpha=0.25
                )

                plt.xticks(
                    rotation=45
                )

                plt.tight_layout()

                st.pyplot(fig3)

                # -------------------------------------
                # FORECAST TABLE
                # -------------------------------------

                st.subheader(
                    "Forecast Results"
                )

                display_forecast = forecast_df.copy()

                display_forecast["Forecast"] = (
                    display_forecast["Forecast"]
                    .round(2)
                )

                st.dataframe(
                    display_forecast,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------
                # DOWNLOAD
                # -------------------------------------

                csv = forecast_df.to_csv(
                    index=False
                )

                st.download_button(
                    "📥 Download Forecast CSV",
                    data=csv,
                    file_name=(
                        "reliance_stock_forecast.csv"
                    ),
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Forecasting failed: {e}"
                )


# =========================================================
# TAB 4 — DATA EXPLORER
# =========================================================

with tab4:

    st.subheader(
        "📋 Stock Data Explorer"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Filtered Dataset",
        data=csv_data,
        file_name="reliance_filtered_data.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Reliance Industries Stock Intelligence Dashboard "
    "• Time Series Analysis • ARIMA • SARIMA • Holt-Winters"
)
