import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import openai

st.set_page_config(page_title="Aussie Housing AI", layout="centered")
st.title("🏡 Australian Housing Market AI Assistant")

openai.api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

st.sidebar.header("Step 1: Upload CSV")
file = st.sidebar.file_uploader("Upload a housing data CSV", type="csv")

if file:
    data = pd.read_csv(file, parse_dates=["date"])
    st.success("✅ Data loaded successfully!")
else:
    st.warning("⚠️ Upload a CSV with columns: suburb, date, median_price, rental_yield, vacancy_rate")
    data = pd.DataFrame(columns=["suburb", "date", "median_price", "rental_yield", "vacancy_rate"])

st.subheader("💬 Ask a question")
prompt = st.text_input("e.g. Show suburbs under 800k with yield over 4% and low vacancy")

def extract_filters_from_prompt(prompt):
    if not prompt.strip() or not openai.api_key:
        return 700000, 4.0, 1.0
    try:
        messages = [
            {
                "role": "system",
                "content": "Extract numeric filter values from housing market questions. Respond only with JSON format: {'max_price': 700000, 'min_yield': 4.0, 'max_vacancy': 1.0}"
            },
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        filters = eval(response['choices'][0]['message']['content'])
        return filters.get("max_price", 700000), filters.get("min_yield", 4.0), filters.get("max_vacancy", 1.0)
    except:
        st.error("⚠️ Could not extract filters. Using defaults.")
        return 700000, 4.0, 1.0

max_price, min_yield, max_vacancy = extract_filters_from_prompt(prompt)

def filter_suburbs(data, max_price, min_yield, max_vacancy):
    if data.empty:
        return pd.DataFrame()
    latest = data['date'].max()
    latest_data = data[data['date'] == latest]
    return latest_data[
        (latest_data['median_price'] <= max_price) &
        (latest_data['rental_yield'] >= min_yield) &
        (latest_data['vacancy_rate'] <= max_vacancy)
    ][['suburb', 'median_price', 'rental_yield', 'vacancy_rate']]

filtered = filter_suburbs(data, max_price, min_yield, max_vacancy)

if not filtered.empty:
    st.subheader("📍 Matching Suburbs")
    st.dataframe(filtered)
else:
    st.info("No suburbs found for the given criteria.")

if 'Swan View' in data['suburb'].unique():
    st.subheader("📈 Forecast for Swan View")
    swan_data = data[data['suburb'] == 'Swan View'][['date', 'median_price']].copy()
    swan_data.rename(columns={"date": "ds", "median_price": "y"}, inplace=True)

    model = Prophet()
    model.fit(swan_data)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title("🏡 Forecast: Swan View Median Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (AUD)")
    plt.grid(True)
    st.pyplot(fig)

    latest = swan_data['y'].iloc[-1]
    future_val = forecast['yhat'].iloc[-1]
    pct_change = (future_val - latest) / latest * 100

    st.markdown(f"**Current Price:** ${latest:,.0f}")
    st.markdown(f"**Predicted (12 mo):** ${future_val:,.0f}")
    st.markdown(f"**Expected Growth:** {pct_change:.2f}%")
else:
    st.info("No Swan View data available for forecasting.")
