import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import io
import plotly.express as px
from geopy.geocoders import Nominatim 
from streamlit_searchbox import st_searchbox

# --- ১. অ্যাপ সেটআপ ---
st.set_page_config(page_title="Energy AI Dashboard", layout="wide")

# --- ২. লগইন ফাংশন (HTML ছাড়া একদম সাধারণ পদ্ধতি) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def check_login():
    if not st.session_state['logged_in']:
        # HTML বাদ দিয়ে সরাসরি Streamlit header ব্যবহার
        st.title("🔐 Energy AI Login")
        
        # ফরমের ভেতরে রাখা হয়েছে যাতে রেন্ডারিং সহজ হয়
        with st.form("login_gate"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter Dashboard", use_container_width=True)
            
            if submitted:
                if user == "abdul" and pw == "123":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        return False
    return True

# লগইন চেক
if not check_login():
    st.stop()

# --- ৩. সাহায্যকারী ফাংশনসমূহ ---
def search_location(searchterm: str):
    if not searchterm or len(searchterm) < 3: return []
    try:
        geolocator = Nominatim(user_agent="energy_app_v2")
        locations = geolocator.geocode(searchterm, exactly_one=False, limit=5)
        return [(loc.address, (loc.latitude, loc.longitude)) for loc in locations] if locations else []
    except: return []

def parse_weather(res):
    return pd.DataFrame({
        "ds": pd.date_range(start=pd.to_datetime(res.Hourly().Time(), unit="s", utc=True), 
                          end=pd.to_datetime(res.Hourly().TimeEnd(), unit="s", utc=True), 
                          freq="H", inclusive="left"),
        "temp": res.Hourly().Variables(0).ValuesAsNumpy(),
        "rain": res.Hourly().Variables(1).ValuesAsNumpy(),
        "humidity": res.Hourly().Variables(2).ValuesAsNumpy()
    })

# --- ৪. মেইন ড্যাশবোর্ড ইন্টারফেস ---
st.sidebar.title("Settings")
if st.sidebar.button("Logout"):
    st.session_state['logged_in'] = False
    st.rerun()

st.title("📊 Energy Forecasting Dashboard")

# ফাইল আপলোড
uploaded_file = st.sidebar.file_uploader("Upload Energy Excel", type=['xlsx'])

if uploaded_file:
    try:
        df_all = pd.read_excel(uploaded_file)
        device_list = sorted(df_all['Devicedescription'].unique().tolist())
        selected_devices = st.sidebar.multiselect("Select Devices", device_list)
        
        # লোকেশন এবং ডেট সেটিংস
        st.sidebar.subheader("📍 Location & Date")
        selected_location = st_searchbox(search_location, key="loc_search", placeholder="Search city (e.g. Milan)")
        lat, lon = (45.2192, 12.2796) 
        if selected_location: lat, lon = selected_location
        
        start_date = st.sidebar.date_input("Training Start", value=datetime(2025, 1, 1))
        forecast_days = st.sidebar.slider("Forecast Days", 7, 30, 15)

        if st.button("🚀 Start AI Analysis", use_container_width=True):
            if not selected_devices:
                st.warning("Please select at least one device.")
            else:
                # আবহাওয়া ডেটা
                status = st.status("Processing...")
                status.write("☁️ Fetching weather data...")
                
                cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)
                
                p = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]}
                df_p = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {**p, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": datetime.now().strftime('%Y-%m-%d')})[0]
                df_f = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", {**p, "forecast_days": 16})[0]
                
                all_weather = pd.concat([parse_weather(df_p), parse_weather(df_f)]).drop_duplicates('ds')
                all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)

                # ডিভাইস লুপ
                for device in selected_devices:
                    st.header(f"Device: {device}")
                    
                    df_d = df_all[df_all['Devicedescription'] == device].copy()
                    df_d = df_d.rename(columns={'Date_Hour': 'ds', 'Amount': 'y'})
                    df_d['ds'] = pd.to_datetime(df_d['ds']).dt.tz_localize(None)

                    # মিসিং ডেটা ট্র্যাকার
                    df_d['date_only'] = df_d['ds'].dt.date
                    day_counts = df_d.groupby('date_only').size()
                    incomplete = day_counts[day_counts < 24]
                    if not incomplete.empty:
                        with st.expander("🔍 Data Gaps Found"):
                            st.write(pd.DataFrame({"Missing Date": incomplete.index, "Missing Hours": 24-incomplete.values}))

                    # Interpolation & Prophet
                    status.write(f"🤖 Training AI for {device}...")
                    full_range = pd.date_range(start=df_d['ds'].min(), end=df_d['ds'].max(), freq='H')
                    df_d = pd.merge(pd.DataFrame({'ds': full_range}), df_d, on='ds', how='left')
                    df_d['y'] = df_d['y'].interpolate().clip(lower=0).ffill().bfill()

                    df_train = pd.merge(df_d, all_weather, on='ds', how='inner')
                    
                    model = Prophet(seasonality_mode='multiplicative')
                    for reg in ['temp', 'rain', 'humidity']: model.add_regressor(reg)
                    model.fit(df_train)

                    # Forecast
                    future = model.make_future_dataframe(periods=forecast_days*24, freq='h')
                    future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left').ffill().bfill()
                    forecast = model.predict(future)
                    
                    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

                    # Accuracy কার্ড
                    df_cv = cross_validation(model, initial='100 days', period='30 days', horizon='14 days')
                    df_metrics = performance_metrics(df_cv)
                    acc = max(0, (1 - df_metrics['smape'].mean()) * 100)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{acc:.1f}%")
                    c2.metric("Avg Error (MAE)", f"{df_metrics['mae'].mean():.2f}")
                    c3.metric("RMSE", f"{df_metrics['rmse'].mean():.2f}")

                status.update(label="✅ Analysis Complete!", state="complete")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload an Excel file to start.")
