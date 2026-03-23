import streamlit as st
import pandas as pd
from prophet import Prophet
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import io
from geopy.geocoders import Nominatim 
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")
st.title("⚡ Energy & Weather Intelligence System (CSV Version)")

# 
st.sidebar.header(" Location Settings")
lat = st.sidebar.number_input("Latitude", value=45.2192, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=12.2796, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("🗓️ Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date (Past Data)", value=datetime.now() - timedelta(days=2))

if st.button("🚀 Run Analysis & Generate Report"):
    try:
        with st.spinner('Processing CSV Data... Please wait.'):
            
            # ১. location name and link create 
            geolocator = Nominatim(user_agent="energy_app")
            location_data = geolocator.reverse(f"{lat}, {lon}", language='en')
            address = location_data.address if location_data else "Unknown Location"
            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"

            # read csv file 
            try:
                df_energy = pd.read_csv("energy_data.csv")
            except FileNotFoundError:
                st.error(" 'energy_data.csv' file not found....")
                st.stop()

            # data cleaning 
            df_energy = df_energy.rename(columns={'Date_Hour': 'ds', 'Amount': 'y'})
            df_energy['ds'] = pd.to_datetime(df_energy['ds']).dt.tz_localize(None)
            
            # data filter 
            mask = (df_energy['ds'] >= pd.Timestamp(start_date)) & (df_energy['ds'] <= pd.Timestamp(end_date))
            df_energy = df_energy.loc[mask].dropna()

            #  Weather API call
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            arch_res = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon, "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'), "hourly": ["temperature_2m", "rain", "relative_humidity_2m"]
            })[0]
            
            fcast_res = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "rain", "relative_humidity_2m"], "forecast_days": 16
            })[0]

            def process_weather(res):
                return pd.DataFrame({
                    "ds": pd.date_range(start=pd.to_datetime(res.Hourly().Time(), unit="s", utc=True),
                                        end=pd.to_datetime(res.Hourly().TimeEnd(), unit="s", utc=True),
                                        freq=pd.Timedelta(seconds=res.Hourly().Interval()), inclusive="left"),
                    "temp": res.Hourly().Variables(0).ValuesAsNumpy(),
                    "rain": res.Hourly().Variables(1).ValuesAsNumpy(),
                    "humidity": res.Hourly().Variables(2).ValuesAsNumpy()
                })

            all_weather = pd.concat([process_weather(arch_res), process_weather(fcast_res)]).drop_duplicates(subset=['ds'])
            all_weather['ds'] = all_weather['ds'].dt.tz_localize(None)

            # model traing and prediction
            df_train = pd.merge(df_energy, all_weather, on='ds', how='inner')
            
            if df_train.empty:
                st.warning(" no data found in this date. pls check...")
                st.stop()

            model = Prophet(weekly_seasonality=True, daily_seasonality=True)
            model.add_country_holidays(country_name='IT')
            model.add_regressor('temp'); model.add_regressor('rain'); model.add_regressor('humidity')
            model.fit(df_train)

            future = model.make_future_dataframe(periods=16*24, freq='h')
            future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left')
            future[['temp', 'rain', 'humidity']] = future[['temp', 'rain', 'humidity']].ffill().bfill()
            forecast = model.predict(future)
            
            #  result
            st.success(" Forecast Ready from CSV!")
            st.link_button(f"📍 Location: {address}", google_maps_url)
            st.divider() 

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Energy Trend Graph")
                st.pyplot(model.plot(forecast))
            with col2:
                st.subheader("Seasonality Analysis")
                st.pyplot(model.plot_components(forecast))

            #  excel download link button 
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                forecast[['ds', 'yhat', 'temp', 'rain', 'humidity']].to_excel(writer, index=False, sheet_name='Forecast_Results')
                all_weather.to_excel(writer, index=False, sheet_name='Weather_Reference')
            
            st.download_button(
                label="📥 Download Full Excel Report",
                data=output.getvalue(),
                file_name=f'Forecast_Report_{datetime.now().strftime("%Y%m%d")}.xlsx',
                mime="application/vnd.ms-excel"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
