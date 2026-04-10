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

# --- ১. পেজ কনফিগারেশন ---
st.set_page_config(page_title="AI Energy & Weather Dashboard", layout="wide")

# --- ২. লগইন ফাংশন (নিরাপদ পদ্ধতি) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def check_login():
    if not st.session_state['logged_in']:
        st.markdown("<h1 style='text-align: center;'>🔐 Energy AI Login</h1>", unsafe_allow_index=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    if user == "admin" and pw == "admin123":
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")
        return False
    return True

# লগইন চেক করা হচ্ছে
if not check_login():
    st.stop()

# --- ৩. সাহায্যকারী ফাংশন (Location & Weather) ---
def search_location(searchterm: str):
    if not searchterm or len(searchterm) < 3: return []
    try:
        geolocator = Nominatim(user_agent="energy_app")
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

# --- ৪. মেইন ড্যাশবোর্ড ---
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Energy Excel (Devicedescription, Date_Hour, Amount)", type=['xlsx'])

if uploaded_file:
    df_all = pd.read_excel(uploaded_file)
    device_list = sorted(df_all['Devicedescription'].unique().tolist())
    selected_devices = st.sidebar.multiselect("Select Device(s)", device_list)
else:
    st.info("👋 Welcome! Please upload your Excel file from the sidebar to start.")
    selected_devices = []

st.sidebar.subheader("📍 Location Settings")
selected_location = st_searchbox(search_location, key="loc_search", placeholder="Search city...")
lat, lon = (45.2192, 12.2796) 
if selected_location: lat, lon = selected_location

start_date = st.sidebar.date_input("Training Start Date", value=datetime(2024, 1, 1))
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 16)

# --- ৫. মূল প্রসেসিং লজিক ---
if st.button("🚀 Run Full Analysis", use_container_width=True):
    if not selected_devices or uploaded_file is None:
        st.warning("Please upload data and select devices!")
        st.stop()

    try:
        # Step A: Weather Data Fetching
        with st.spinner('⌛ Fetching Weather Data...'):
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            p = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]}
            df_p_raw = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {**p, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": datetime.now().strftime('%Y-%m-%d')})[0]
            df_f_raw = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", {**p, "forecast_days": 16})[0]
            
            all_weather = pd.concat([parse_weather(df_p_raw), parse_weather(df_f_raw)]).drop_duplicates('ds')
            all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)

        st.subheader("☁️ Weather Trend Insight")
        st.plotly_chart(px.line(all_weather.tail(24*16), x='ds', y=['temp', 'rain', 'humidity']), use_container_width=True)

        # Step B: Device Processing
        all_forecasts = []
        for device in selected_devices:
            st.markdown(f"---")
            st.markdown(f"## 📊 Analysis for: {device}")
            
            df_device = df_all[df_all['Devicedescription'] == device].copy()
            df_device = df_device.rename(columns={'Date_Hour': 'ds', 'Amount': 'y'})
            df_device['ds'] = pd.to_datetime(df_device['ds']).dt.tz_localize(None)

            # Missing Data Tracking (আপনার SQL HAVING COUNT লজিক)
            df_device['date_only'] = df_device['ds'].dt.date
            day_counts = df_device.groupby('date_only').size()
            incomplete_days = day_counts[day_counts < 24]
            
            if not incomplete_days.empty:
                with st.expander(f"🔍 Data Gaps Found ({len(incomplete_days)} days)", expanded=False):
                    gap_df = pd.DataFrame({"Date": incomplete_days.index, "Found Hours": incomplete_days.values, "Missing": 24-incomplete_days.values})
                    st.table(gap_df)

            # Interpolation & Training
            full_range = pd.date_range(start=df_device['ds'].min(), end=df_device['ds'].max(), freq='H')
            df_device = pd.merge(pd.DataFrame({'ds': full_range}), df_device, on='ds', how='left')
            df_device['y'] = df_device['y'].interpolate(method='linear').clip(lower=0).ffill().bfill()

            df_train = pd.merge(df_device, all_weather, on='ds', how='inner')

            with st.spinner(f"Training AI for {device}..."):
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, 
                               seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
                model.add_country_holidays(country_name='IT')
                for reg in ['temp', 'rain', 'humidity']: model.add_regressor(reg)
                model.fit(df_train)

                # Prediction
                future = model.make_future_dataframe(periods=forecast_days*24, freq='h')
                future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left').ffill().bfill()
                forecast = model.predict(future)
                forecast['yhat'] = forecast['yhat'].clip(lower=0)

                # Charts
                st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

                # Accuracy (Cross Validation)
                st.subheader(f"🎯 Accuracy Metrics")
                df_cv = cross_validation(model, initial='180 days', period='30 days', horizon='14 days')
                df_metrics = performance_metrics(df_cv)
                acc = max(0, (1 - df_metrics['smape'].mean()) * 100)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Model Accuracy", f"{acc:.1f}%")
                c2.metric("Avg Error (MAE)", f"{df_metrics['mae'].mean():.2f}")
                c3.metric("RMSE", f"{df_metrics['rmse'].mean():.2f}")

                report = forecast[['ds', 'yhat']].copy()
                report['Device'] = device
                all_forecasts.append(report)

        if all_forecasts:
            st.session_state['final_data'] = pd.concat(all_forecasts)
            st.success("✅ Full Analysis Completed!")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# --- ৬. এক্সপোর্ট অপশন ---
if 'final_data' in st.session_state and st.session_state['final_data'] is not None:
    st.divider()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        st.session_state['final_data'].to_excel(writer, index=False, sheet_name='Forecast_Report')
    st.download_button("📥 Download Excel Report", data=output.getvalue(), file_name="Forecast_Report.xlsx", mime="application/vnd.ms-excel")
