import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import plotly.express as px
from geopy.geocoders import Nominatim 
import time

# --- Page Config ---
st.set_page_config(page_title="AI Energy Predictor Pro", layout="wide")

# --- 1. Login Page Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>🔐 Energy AI Login</h2>
        </div>
    """, unsafe_allow_index=True)
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pw == "admin123": # আপনার পছন্দমতো পাসওয়ার্ড দিন
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state['logged_in']:
    login()
    st.stop()

# --- 2. Main App Content ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    </style>
    """, unsafe_allow_index=True)

st.title("📊 AI Energy Forecasting & Missing Data Analytics")

# --- 3. Sidebar Data Loading ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Energy Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # Column Settings
    st.sidebar.subheader("Column Mapping")
    device_col = st.sidebar.selectbox("Device Name", df_raw.columns)
    date_col = st.sidebar.selectbox("Date/Hour Column", df_raw.columns)
    value_col = st.sidebar.selectbox("Energy/Amount Column", df_raw.columns)

    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    available_devices = df_raw[device_col].unique()
    selected_devices = st.sidebar.multiselect("Select Devices", available_devices)

    # Model Settings
    u_yearly = st.sidebar.checkbox("Yearly Seasonality", value=True)
    u_weekly = st.sidebar.checkbox("Weekly Seasonality", value=True)
    u_daily = st.sidebar.checkbox("Daily Seasonality", value=True)
    u_seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'])
    u_changepoint = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
    forecast_days = st.sidebar.number_input("Forecast Days", 1, 365, 15)

    if st.button("🚀 Run Full Analysis"):
        if not selected_devices:
            st.warning("Please select a device!")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # 4. Weather Fetching
        status_text.markdown("✨ **Step 1/4:** Fetching Weather Data...")
        lat, lon = 45.4642, 9.1900 # Default Italy
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        p = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]}
        start_date_p = df_raw[date_col].min()
        
        res_p = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {
            **p, "start_date": start_date_p.strftime('%Y-%m-%d'), 
            "end_date": datetime.now().strftime('%Y-%m-%d')
        })[0]

        def parse_w(res):
            return pd.DataFrame({
                "ds": pd.date_range(start=pd.to_datetime(res.Hourly().Time(), unit="s", utc=True), 
                                  end=pd.to_datetime(res.Hourly().TimeEnd(), unit="s", utc=True), 
                                  freq="H", inclusive="left"),
                "temp": res.Hourly().Variables(0).ValuesAsNumpy(),
                "rain": res.Hourly().Variables(1).ValuesAsNumpy(),
                "humidity": res.Hourly().Variables(2).ValuesAsNumpy()
            })
        
        all_weather = parse_w(res_p)
        all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)
        progress_bar.progress(25)

        # 5. Missing Data Tracking Part
        status_text.markdown("🔍 **Step 2/4:** Tracking Missing/Bad Data...")
        missing_reports = []
        for device in selected_devices:
            df_m = df_raw[df_raw[device_col] == device].copy()
            df_m['date_only'] = df_m[date_col].dt.date
            daily_counts = df_m.groupby('date_only').size()
            incomplete_days = daily_counts[daily_counts < 24]
            if not incomplete_days.empty:
                for d, count in incomplete_days.items():
                    missing_reports.append({"Device": device, "Date": d, "Hours Found": count, "Missing": 24-count})
        
        if missing_reports:
            with st.expander("🔍 Missed/Bad Data Tracking Report", expanded=False):
                st.table(pd.DataFrame(missing_reports))
        progress_bar.progress(40)

        # 6. Training & Forecast Loop
        for device in selected_devices:
            status_text.markdown(f"🔌 **Step 3/4:** Training Model for {device}...")
            df_d = df_raw[df_raw[device_col] == device].rename(columns={date_col: 'ds', value_col: 'y'}).copy()
            df_d['ds'] = df_d['ds'].dt.tz_localize(None)

            # Interpolation
            full_range = pd.date_range(start=df_d['ds'].min(), end=df_d['ds'].max(), freq='H')
            df_d = pd.merge(pd.DataFrame({'ds': full_range}), df_d, on='ds', how='left')
            df_d['y'] = df_d['y'].interpolate(method='linear').clip(lower=0).ffill().bfill()

            df_train = pd.merge(df_d, all_weather, on='ds', how='inner')

            model = Prophet(yearly_seasonality=u_yearly, weekly_seasonality=u_weekly, daily_seasonality=u_daily,
                           seasonality_mode=u_seasonality_mode, changepoint_prior_scale=u_changepoint)
            model.add_country_holidays(country_name='IT')
            for reg in ['temp', 'rain', 'humidity']: model.add_regressor(reg)
            
            model.fit(df_train)

            # Future Forecast
            future = model.make_future_dataframe(periods=forecast_days*24, freq='H')
            future = pd.merge(future, all_weather, on='ds', how='left')
            for reg in ['temp', 'rain', 'humidity']: future[reg] = future[reg].fillna(all_weather[reg].mean())
            
            forecast = model.predict(future)
            
            # --- Results Display ---
            st.markdown(f"## 📊 {device} Results")
            st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
            
            # 7. Cross Validation Part
            status_text.markdown(f"🧪 **Step 4/4:** Validating Model for {device}...")
            with st.spinner("Running Cross-Validation..."):
                df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='15 days')
                df_metrics = performance_metrics(df_cv)
                
                # Accuracy Cards
                st.markdown("### 🎯 Accuracy Summary")
                acc_pct = max(0, (1 - df_metrics['smape'].mean()) * 100)
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Accuracy", f"{acc_pct:.1f}%")
                c2.metric("MAE (Avg Error)", f"{df_metrics['mae'].mean():.2f} kWh")
                c3.metric("RMSE", f"{df_metrics['rmse'].mean():.2f} kWh")

        progress_bar.progress(100)
        status_text.success("✅ Full Analysis Completed!")

else:
    st.info("👋 Welcome! Please upload your Excel file to begin.")
