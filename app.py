import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
st.set_page_config(page_title="Reliance Industries Forecast",page_icon="📈",layout="wide")
st.markdown("<h1 style='text-align:center;'>📈 Reliance Industries Stock Forecasting Dashboard</h1>",unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Advanced Time-Series Forecasting | ARIMA / SARIMA / Holt-Winters / Auto-ARIMA</p>",unsafe_allow_html=True)
@st.cache_data
def load_data():
    df=pd.read_csv("data/reliance_stock_data.csv")
    df["Date"]=pd.to_datetime(df["Date"])
    df=df.sort_values("Date").drop_duplicates("Date")
    return df
@st.cache_resource
def load_model():
    with open("saved_model/reliance_best_model.pkl","rb") as file:
        return pickle.load(file)
df=load_data()
model=load_model()
df["MA20"]=df["Close"].rolling(20).mean()
df["MA50"]=df["Close"].rolling(50).mean()
df["MA200"]=df["Close"].rolling(200).mean()
df["Returns"]=df["Close"].pct_change()*100
df["Volatility"]=df["Returns"].rolling(20).std()
st.sidebar.title("⚙️ Dashboard Controls")
start_date=st.sidebar.date_input("Start Date",df["Date"].min().date())
end_date=st.sidebar.date_input("End Date",df["Date"].max().date())
forecast_days=st.sidebar.slider("Forecast Horizon",1,30,30)
show_ma20=st.sidebar.checkbox("MA20",True)
show_ma50=st.sidebar.checkbox("MA50",True)
show_ma200=st.sidebar.checkbox("MA200",True)
filtered_df=df[(df["Date"]>=pd.to_datetime(start_date))&(df["Date"]<=pd.to_datetime(end_date))]
last_price=df["Close"].iloc[-1]
highest_price=df["Close"].max()
lowest_price=df["Close"].min()
avg_price=df["Close"].mean()
c1,c2,c3,c4=st.columns(4)
c1.metric("Latest Close",f"₹{last_price:.2f}")
c2.metric("Highest Price",f"₹{highest_price:.2f}")
c3.metric("Lowest Price",f"₹{lowest_price:.2f}")
c4.metric("Average Price",f"₹{avg_price:.2f}")
st.divider()
st.subheader("📈 Historical Price Analysis")
fig,ax=plt.subplots(figsize=(14,5))
ax.plot(filtered_df["Date"],filtered_df["Close"],label="Close Price")
if show_ma20:
    ax.plot(filtered_df["Date"],filtered_df["MA20"],label="MA20")
if show_ma50:
    ax.plot(filtered_df["Date"],filtered_df["MA50"],label="MA50")
if show_ma200:
    ax.plot(filtered_df["Date"],filtered_df["MA200"],label="MA200")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Reliance Industries Price and Moving Averages")
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.subheader("📊 Returns and Volatility")
c1,c2=st.columns(2)
with c1:
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(filtered_df["Date"],filtered_df["Returns"])
    ax.set_title("Daily Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return %")
    ax.grid(True)
    st.pyplot(fig)
with c2:
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(filtered_df["Date"],filtered_df["Volatility"])
    ax.set_title("20-Day Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    st.pyplot(fig)
st.divider()
st.subheader("🔮 Future Forecast")
try:
    forecast=model.predict(n_periods=forecast_days)
except:
    forecast=model.forecast(steps=forecast_days)
forecast=np.asarray(forecast,dtype=float)
future_dates=pd.bdate_range(start=df["Date"].iloc[-1]+pd.Timedelta(days=1),periods=forecast_days)
forecast_df=pd.DataFrame({"Date":future_dates,"Forecast Price":forecast})
forecast_last=forecast[-1]
forecast_change=((forecast_last-last_price)/last_price)*100
c1,c2,c3=st.columns(3)
c1.metric("Current Price",f"₹{last_price:.2f}")
c2.metric("Final Forecast",f"₹{forecast_last:.2f}")
c3.metric("Forecast Change",f"{forecast_change:.2f}%")
fig,ax=plt.subplots(figsize=(14,6))
ax.plot(df["Date"].tail(252),df["Close"].tail(252),label="Historical Price")
ax.plot(forecast_df["Date"],forecast_df["Forecast Price"],label="Forecast")
ax.axvline(df["Date"].iloc[-1],linestyle="--",label="Forecast Start")
ax.set_title(f"Reliance Industries - Next {forecast_days} Business Days")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.subheader("📋 Forecast Table")
display_forecast=forecast_df.copy()
display_forecast["Forecast Price"]=display_forecast["Forecast Price"].round(2)
st.dataframe(display_forecast,use_container_width=True)
csv=forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Forecast CSV",csv,"reliance_30_day_forecast.csv","text/csv")
st.divider()
st.subheader("📊 Model Performance")
try:
    results=pd.read_csv("model_results.csv")
    required=["Model","MAE","MSE","RMSE","R2 Score","MAPE"]
    if all(x in results.columns for x in required):
        st.dataframe(results,use_container_width=True)
        fig,ax=plt.subplots(figsize=(12,5))
        ax.bar(results["Model"],results["RMSE"])
        ax.set_title("Model Comparison - RMSE")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x",rotation=30)
        ax.grid(axis="y")
        st.pyplot(fig)
except:
    st.info("Model performance file not found. Add model_results.csv after model evaluation.")
st.subheader("📉 Recent Price Movement")
recent=df.tail(30)[["Date","Close","MA20","MA50","MA200"]].copy()
recent["Close"]=recent["Close"].round(2)
recent["MA20"]=recent["MA20"].round(2)
recent["MA50"]=recent["MA50"].round(2)
recent["MA200"]=recent["MA200"].round(2)
st.dataframe(recent,use_container_width=True)
st.divider()
st.warning("⚠️ This dashboard provides model-based statistical forecasts. Stock prices are influenced by market conditions, news, economic factors and unexpected events. Forecasts should not be treated as guaranteed future prices.")
st.markdown("<p style='text-align:center;'>Reliance Industries Time-Series Forecasting Project</p>",unsafe_allow_html=True)