import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import plotly.express as px

# --- ১. কনফিগারেশন এবং স্টাইলিং (Python 3.14 Safe Version) ---
st.set_page_config(page_title="AI Energy Predictor Pro", layout="wide")

def local_css():
    # স্টাইলকে ছোট ছোট ব্লকে ভাগ করা হয়েছে যাতে রেন্ডারিং এরর না হয়
    style = """
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    .main-card { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; }
    </style>
    """
    st.markdown(style, unsafe_allow_index=True)

# --- ২. লগইন লজিক ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    local_css()
    # Container ব্যবহার করা হয়েছে TypeError এড়ানোর জন্য
    with st.container():
        st.markdown("<h1 style='text-align: center;'>🔐 Energy AI Login</h1>", unsafe_allow_index=True)
        user = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        
        if st.button("Login", use_container_width=True):
            if user == "admin" and pw == "admin123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid Credentials!")

# --- ৩. মেইন কন্ট্রোল ফ্লো ---
if not st.session_state['logged_in']:
    login_page()
    st.stop()

# --- ৪. মেইন অ্যাপ (লগইন হওয়ার পর) ---
local_css()
with st.sidebar:
    st.title("Settings")
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    # ডেটা লোড
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        # কলাম সিলেকশন
        col1, col2, col3 = st.columns(3)
        with col1: device_col = st.selectbox("Device Name Column", df_raw.columns)
        with col2: date_col = st.selectbox("Date Column", df_raw.columns)
        with col3: value_col = st.selectbox("Value Column", df_raw.columns)

        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        selected_devices = st.multiselect("Select Devices", df_raw[device_col].unique())

        if st.button("🚀 Run Analysis"):
            if not selected_devices:
                st.warning("Please select a device.")
            else:
                # --- ৫. আবহাওয়া ডেটা ---
                lat, lon = 45.4642, 9.1900 
                cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)
                
                start_date_p = df_raw[date_col].min().strftime('%Y-%m-%d')
                res_p = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {
                    "latitude": lat, "longitude": lon, 
                    "start_date": start_date_p, 
                    "end_date": datetime.now().strftime('%Y-%m-%d'),
                    "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]
                })[0]

                # আবহাওয়া ডেটা পার্সিং
                all_weather = pd.DataFrame({
                    "ds": pd.date_range(start=pd.to_datetime(res_p.Hourly().Time(), unit="s", utc=True), 
                                      end=pd.to_datetime(res_p.Hourly().TimeEnd(), unit="s", utc=True), 
                                      freq="H", inclusive="left"),
                    "temp": res_p.Hourly().Variables(0).ValuesAsNumpy(),
                    "rain": res_p.Hourly().Variables(1).ValuesAsNumpy(),
                    "humidity": res_p.Hourly().Variables(2).ValuesAsNumpy()
                })
                all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)

                # --- ৬. মিসিং ডেটা ট্র্যাকার ---
                missing_reports = []
                for device in selected_devices:
                    df_m = df_raw[df_raw[device_col] == device].copy()
                    df_m['date_only'] = df_m[date_col].dt.date
                    counts = df_m.groupby('date_only').size()
                    for d, c in counts[counts < 24].items():
                        missing_reports.append({"Device": device, "Date": d, "Missing Hours": 24-c})
                
                if missing_reports:
                    with st.expander("🔍 Data Integrity Report"):
                        st.table(pd.DataFrame(missing_reports))

                # --- ৭. ট্রেনিং এবং ফোরকাস্টিং ---
                for device in selected_devices:
                    st.subheader(f"📊 Analysis for {device}")
                    df_d = df_raw[df_raw[device_col] == device].rename(columns={date_col: 'ds', value_col: 'y'}).copy()
                    df_d['ds'] = df_d['ds'].dt.tz_localize(None)

                    # Interpolation
                    full_range = pd.date_range(start=df_d['ds'].min(), end=df_d['ds'].max(), freq='H')
                    df_d = pd.merge(pd.DataFrame({'ds': full_range}), df_d, on='ds', how='left')
                    df_d['y'] = df_d['y'].interpolate(method='linear').clip(lower=0).ffill().bfill()

                    df_train = pd.merge(df_d, all_weather, on='ds', how='inner')

                    model = Prophet()
                    for reg in ['temp', 'rain', 'humidity']: model.add_regressor(reg)
                    model.fit(df_train)

                    future = model.make_future_dataframe(periods=24*15, freq='H')
                    future = pd.merge(future, all_weather, on='ds', how='left').ffill().bfill()
                    
                    forecast = model.predict(future)
                    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

                    # Cross Validation
                    df_cv = cross_validation(model, initial='100 days', period='30 days', horizon='15 days')
                    df_m = performance_metrics(df_cv)
                    acc = max(0, (1 - df_m['smape'].mean()) * 100)
                    st.metric("Model Accuracy", f"{acc:.1f}%")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👋 Welcome! Please upload your Excel file from the sidebar.")
