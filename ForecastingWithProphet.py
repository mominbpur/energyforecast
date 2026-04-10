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
import time

# --- ১. কনফিগারেশন এবং স্টাইলিং ---
st.set_page_config(page_title="AI Energy Predictor Pro", layout="wide")

def local_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    .main-card { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px; }
    .metric-box { background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_index=True)

# --- ২. লগইন লজিক ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    local_css()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 50px;'><h1>🔐 Energy AI Login</h1></div>", unsafe_allow_index=True)
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if user == "admin" and pw == "admin123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid Credentials!")

if not st.session_state['logged_in']:
    login_page()
    st.stop()

# --- ৩. মেইন অ্যাপ (লগইন হওয়ার পর) ---
local_css()
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
st.title("📊 AI Energy Forecasting & Missing Data Tracker")

# --- ৪. ডেটা আপলোড এবং কলাম সিলেকশন ---
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=["xlsx", "csv"])

if uploaded_file:
    # ডেটা লোড
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # কলাম ম্যাপিং
    st.sidebar.subheader("Select Columns")
    device_col = st.sidebar.selectbox("Device Name Column", df_raw.columns)
    date_col = st.sidebar.selectbox("Date/Time Column", df_raw.columns)
    value_col = st.sidebar.selectbox("Energy Value Column", df_raw.columns)

    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    available_devices = df_raw[device_col].unique()
    selected_devices = st.sidebar.multiselect("Select Devices to Analyze", available_devices)

    # ফোরকাস্টিং সেটিংস
    st.sidebar.subheader("🤖 Model Parameters")
    forecast_days = st.sidebar.number_input("Forecast Days", 1, 365, 15)
    u_seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'])
    u_changepoint = st.sidebar.slider("Changepoint Scale", 0.001, 0.5, 0.05)

    if st.button("🚀 Run Analysis", use_container_width=True):
        if not selected_devices:
            st.warning("Please select at least one device.")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # --- ৫. আবহাওয়া ডেটা (Open-Meteo) ---
        status_text.markdown("☁️ **Step 1/4:** Fetching Weather Regressors...")
        lat, lon = 45.4642, 9.1900 
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        p = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]}
        start_date_p = df_raw[date_col].min()
        res_p = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {
            **p, "start_date": start_date_p.strftime('%Y-%m-%d'), 
            "end_date": (datetime.now() + timedelta(days=16)).strftime('%Y-%m-%d')
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

        # --- ৬. মিসিং ডেটা ট্র্যাকার (Missing Data Tracker) ---
        status_text.markdown("🔍 **Step 2/4:** Analyzing Data Integrity...")
        missing_reports = []
        for device in selected_devices:
            df_m = df_raw[df_raw[device_col] == device].copy()
            df_m['date_only'] = df_m[date_col].dt.date
            counts = df_m.groupby('date_only').size()
            bad_days = counts[counts < 24]
            for d, c in bad_days.items():
                missing_reports.append({"Device": device, "Date": d, "Hours": c, "Missing": 24-c})
        
        if missing_reports:
            with st.expander("🔍 Missing/Bad Data Report", expanded=False):
                st.table(pd.DataFrame(missing_reports))
        progress_bar.progress(50)

        # --- ৭. ট্রেনিং এবং ফোরকাস্টিং লুপ ---
        for device in selected_devices:
            status_text.markdown(f"⚙️ **Step 3/4:** Training Model for {device}...")
            df_d = df_raw[df_raw[device_col] == device].rename(columns={date_col: 'ds', value_col: 'y'}).copy()
            df_d['ds'] = df_d['ds'].dt.tz_localize(None)

            # Interpolation (Missing Data Recovery)
            full_range = pd.date_range(start=df_d['ds'].min(), end=df_d['ds'].max(), freq='H')
            df_d = pd.merge(pd.DataFrame({'ds': full_range}), df_d, on='ds', how='left')
            df_d['y'] = df_d['y'].interpolate(method='linear').clip(lower=0).ffill().bfill()

            # Merge Weather
            df_train = pd.merge(df_d, all_weather, on='ds', how='inner')

            # Prophet Model
            model = Prophet(seasonality_mode=u_seasonality_mode, changepoint_prior_scale=u_changepoint)
            model.add_country_holidays(country_name='IT')
            for reg in ['temp', 'rain', 'humidity']: model.add_regressor(reg)
            model.fit(df_train)

            # Forecast
            future = model.make_future_dataframe(periods=forecast_days*24, freq='H')
            future = pd.merge(future, all_weather, on='ds', how='left')
            for reg in ['temp', 'rain', 'humidity']: 
                future[reg] = future[reg].fillna(all_weather[reg].mean())
            
            forecast = model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)

            # --- ৮. রেজাল্ট এবং একিউরেসি ---
            st.markdown(f"### 📈 Forecast for {device}")
            st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

            status_text.markdown(f"🧪 **Step 4/4:** Validating {device}...")
            # Cross Validation
            df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='15 days')
            df_m = performance_metrics(df_cv)
            
            acc_pct = max(0, (1 - df_m['smape'].mean()) * 100)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🎯 Model Accuracy", f"{acc_pct:.1f}%")
            with c2: st.metric("📏 Avg Error (MAE)", f"{df_m['mae'].mean():.2f} kWh")
            with c3: st.metric("📉 Error Variation (RMSE)", f"{df_m['rmse'].mean():.2f} kWh")
            st.markdown("---")

        progress_bar.progress(100)
        status_text.success("✅ Analysis Complete!")
else:
    st.info("👋 Welcome! Please upload your Excel file from the sidebar to start.")
