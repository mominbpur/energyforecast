import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import io
import plotly.express as px
from geopy.geocoders import Nominatim 
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox

# Page Configuration
st.set_page_config(page_title="AI Energy & Weather Dashboard", layout="wide")
st.title("⚡ Advanced Energy AI & Weather Analytics")
st.markdown("---")

# Session State Initialization
if 'final_data' not in st.session_state: st.session_state['final_data'] = None
if 'raw_energy_data' not in st.session_state: st.session_state['raw_energy_data'] = None

# get lat,lon from location 
def search_location(searchterm: str):
    if not searchterm or len(searchterm) < 3:
        return []
    try:
        geolocator = Nominatim(user_agent="energy_app_v2")
        locations = geolocator.geocode(searchterm, exactly_one=False, limit=5)
        if locations:
            return [(loc.address, (loc.latitude, loc.longitude)) for loc in locations]
    except:
        return []
    return []

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# Excel File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Energy Data (Excel)", type=['xlsx'])

if uploaded_file:
    try:
        df_all = pd.read_excel(uploaded_file)
        st.session_state['raw_energy_data'] = df_all
        device_list = sorted(df_all['Devicedescription'].unique().tolist())
    except Exception as e:
        st.sidebar.error(f"Error reading Excel: {e}")
        device_list = []
else:
    st.sidebar.info("Please upload an Excel file to start.")
    device_list = []

selected_devices = st.sidebar.multiselect("Select Device(s)", device_list)

st.sidebar.subheader(":material/location_on: Location Settings")

# Location search bar 
selected_location = st_searchbox(
    search_location,
    key="location_search",
    placeholder="Search city or address...",
    label="Search Location (Auto-suggest)"
)

lat, lon = 45.2192, 12.2796 
if selected_location:
    lat, lon = selected_location
    st.sidebar.success(f"Selected: {lat}, {lon}")

lat = st.sidebar.number_input("Latitude", value=lat, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=lon, format="%.4f")

start_date = st.sidebar.date_input("Start Date", value=datetime(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2026, 2, 24))

# --- Main Logic ---
if st.button(f"🚀 Run Full Analysis"):
    if not selected_devices or st.session_state['raw_energy_data'] is None:
        st.warning("Please upload data and select at least one device!")
        st.stop()

    try:
        # 1. Weather Data Fetching
        with st.spinner('⌛ Fetching Weather Data...'):
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            def parse_w(res):
                start_dt = pd.to_datetime(res.Hourly().Time(), unit="s", utc=True)
                end_dt = pd.to_datetime(res.Hourly().TimeEnd(), unit="s", utc=True)
                
                # এখানে freq="h" (ছোট হাতের) এবং ইঙ্ক্লুসিভ প্যারামিটারটি চেক করুন
                return pd.DataFrame({
                    "ds": pd.date_range(
                        start=start_dt, 
                        end=end_dt, 
                        freq="h", 
                        inclusive="left"  # নিশ্চিত করুন বানান ঠিক আছে
                    ),
                    "temp": res.Hourly().Variables(0).ValuesAsNumpy(),
                    "rain": res.Hourly().Variables(1).ValuesAsNumpy(),
                    "humidity": res.Hourly().Variables(2).ValuesAsNumpy()
                })

            p = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]}
            df_p = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", {**p, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": datetime.now().strftime('%Y-%m-%d')})[0]
            df_f = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", {**p, "forecast_days": 16})[0]
            
            all_weather = pd.concat([parse_w(df_p), parse_w(df_f)]).drop_duplicates('ds')
            all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)

        st.subheader("☁️ Global Weather Trend (Next 16 Days)")
        st.plotly_chart(px.line(all_weather.tail(24*16), x='ds', y=['temp', 'rain', 'humidity']), use_container_width=True)

        # 2. Device-wise Processing
        all_reports = []
        df_source = st.session_state['raw_energy_data']

        for device in selected_devices:
            st.markdown(f"## 📊 Device: {device}")
            
            df_energy = df_source[df_source['Devicedescription'] == device].copy()
            df_energy = df_energy.rename(columns={'Date_Hour': 'ds', 'Amount': 'y'})
            df_energy['ds'] = pd.to_datetime(df_energy['ds']).dt.tz_localize(None)
            
            mask = (df_energy['ds'].dt.date >= start_date) & (df_energy['ds'].dt.date <= end_date)
            df_energy = df_energy.loc[mask]

            if df_energy.empty:
                st.warning(f"No data for {device}")
                continue
            
        
        df_train = pd.merge(df_energy, all_weather, on='ds', how='inner')
        df_train[['temp', 'rain', 'humidity']] = df_train[['temp', 'rain', 'humidity']].ffill().bfill()

            # Prophet Training
            with st.spinner(f'AI is learning patterns for {device}...'):
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, 
                               changepoint_prior_scale=0.1, seasonality_prior_scale=10.0,
                               interval_width=0.80, seasonality_mode='multiplicative')
                model.add_country_holidays(country_name='IT')
                for reg in ['temp', 'rain', 'humidity']: 
                    model.add_regressor(reg, prior_scale=10.0, mode='multiplicative')
                model.fit(df_train)

            # Performance Metrics
            st.subheader(f"📈 Model Performance Metrics: {device}")
            total_days = (df_train['ds'].max() - df_train['ds'].min()).days
            initial_days = int(total_days * 0.7)
            
            if initial_days > 30:
                df_cv = cross_validation(model, initial=f'{initial_days} days', period='15 days', horizon='14 days')
                df_p_metrics = performance_metrics(df_cv)
                
                # Accuracy Summary
                if not df_p_metrics.empty:
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    avg_mape = df_p_metrics['mape'].mean()
                    accuracy_pct = max(0, (1 - avg_mape) * 100)
                    
                    c1.metric("🎯 Model Accuracy", f"{accuracy_pct:.1f}%")
                    c2.metric("📏 Avg. Error (MAE)", f"{df_p_metrics['mae'].mean():.2f}")
                    c3.metric("📉 Error Variation (RMSE)", f"{df_p_metrics['rmse'].mean():.2f}")
                    st.markdown("---")
                
                st.dataframe(df_p_metrics.head(), use_container_width=True)
            
            # Forecasting
            future = model.make_future_dataframe(periods=16*24, freq='h')
            future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left')
            #future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='').ffill().bfill()
            forecast = model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)

            st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
            
            final_report = forecast[['ds', 'yhat']].merge(df_energy[['ds', 'y']], on='ds', how='')
            final_report = final_report.merge(all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='')
            final_report['Device_Name'] = device
            final_report['Process_Date'] = datetime.now()
            all_reports.append(final_report)

        if all_reports:
            st.session_state['final_data'] = pd.concat(all_reports, ignore_index=True)
            st.sidebar.success("✅ Analysis Complete!")

    except Exception as e:
        st.error(f"Error: {e}")

# Export Logic
if st.session_state['final_data'] is not None:
    st.divider()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        st.session_state['final_data'].to_excel(writer, index=False, sheet_name='Forecast_Report')
    st.download_button("📥 Download Excel Report", output.getvalue(), f"Forecast_{datetime.now().strftime('%Y%m%d')}.xlsx", type="primary")

    # YoY Analysis (Using Uploaded Data)
    st.header("🗓️ Year-over-Year (YoY) Trends")
    df_yoy = st.session_state['raw_energy_data'].copy()
    df_yoy['Year'] = pd.to_datetime(df_yoy['Date_Hour']).dt.year
    df_yoy['Week'] = pd.to_datetime(df_yoy['Date_Hour']).dt.isocalendar().week
    
    for dev in selected_devices:
        dev_yoy = df_yoy[df_yoy['Devicedescription'] == dev]
        fig_yoy = px.line(dev_yoy, x='Date_Hour', y='Amount', color='Year', title=f"YoY Trend: {dev}")
        st.plotly_chart(fig_yoy, use_container_width=True)
