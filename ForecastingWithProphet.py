
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
import seaborn as sns

# graph style
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Multi-Device Energy Forecast", layout="wide")
st.title("⚡ Energy & Weather Intelligence System ")

# data load
@st.cache_data
def load_data():
    try:
        # read excel
        df = pd.read_excel("ed.xlsx") 
        return df
    except Exception as e:
        st.error(f"problem in load file... {e}")
        return None

df_all = load_data()

if df_all is not None:
    # sidebar config
    st.sidebar.header("🔧 Settings")
    
    # device select 
    device_list = sorted(df_all['Devicedescription'].unique().tolist())
    selected_device = st.sidebar.selectbox("Select Device", device_list)
    
    st.sidebar.markdown("---")
    st.sidebar.header(" Location")
    lat = st.sidebar.number_input("Latitude", value=45.2192, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=12.2796, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.header("🗓️ Date Range")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date (Past)", value=datetime.now() - timedelta(days=2))

    # analysis button
    if st.button(f"🚀 Run Forecast for {selected_device}"):
        try:
            with st.spinner(f'Analyzing data for {selected_device}...'):
                
                # filter (selected device and output filter)
                df_energy = df_all[(df_all['Devicedescription'] == selected_device)]
                
                df_energy = df_energy.rename(columns={'Date_Hour': 'ds', 'Amount': 'y'})
                df_energy['ds'] = pd.to_datetime(df_energy['ds']).dt.tz_localize(None)
                
                # date filter from user 
                mask = (df_energy['ds'] >= pd.Timestamp(start_date)) & (df_energy['ds'] <= pd.Timestamp(end_date))
                df_energy = df_energy.loc[mask].dropna()

                if df_energy.empty:
                    st.warning(f" {selected_device}- no data found for this fevice.")
                    st.stop()

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

                # Prophet model training
                df_train = pd.merge(df_energy, all_weather, on='ds', how='inner')
                model = Prophet(weekly_seasonality=True, daily_seasonality=True)
                model.add_country_holidays(country_name='IT')
                model.add_regressor('temp'); model.add_regressor('rain'); model.add_regressor('humidity')
                model.fit(df_train)

                # 16 days weather 
                future = model.make_future_dataframe(periods=16*24, freq='h')
                future = pd.merge(future, all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left')
                future[['temp', 'rain', 'humidity']] = future[['temp', 'rain', 'humidity']].ffill().bfill()
                forecast = model.predict(future)

                forecast['yhat'] = forecast['yhat'].clip(lower=0)
                
                st.success(f" {selected_device} Forecast is Ready!")

                # graph
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Energy Trend Forecast")
                    fig1 = model.plot(forecast)
                    st.pyplot(fig1)
                with col2:
                    st.subheader("Seasonal Patterns")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                # report download  
                output = io.BytesIO()
                ## --------------------------------
                # আসল ডেটা (y) এবং মডেলের প্রেডিকশন (yhat) একসাথে করা হচ্ছে
                final_df = forecast[['ds', 'yhat']].merge(
                    df_energy_filtered[['ds', 'y']], 
                    on='ds', 
                    how='left'
                )
                
                # কলামের নাম সুন্দর করা
                final_df = final_df.rename(columns={'y': 'Original_Data', 'yhat': 'Predicted_Forecast'})
                
                # আবহাওয়ার আসল রিডিং যোগ করা
                final_df = final_df.merge(all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left')
                # --------------------------------
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    #forecast[['ds', 'yhat', 'temp', 'rain', 'humidity']].to_excel(writer, index=False, sheet_name='Forecast_Results')
                    # actual weather temp, rain, humidity
                    final_output = forecast[['ds', 'yhat']].merge(all_weather[['ds', 'temp', 'rain', 'humidity']], on='ds', how='left')
                    # save to excel 
                    final_output.to_excel(writer, index=False, sheet_name='Forecast_Results')
                    
                st.download_button(
                    label=" Download Excel Report",
                    data=output.getvalue(),
                    file_name=f'Forecast_{selected_device}_{datetime.now().strftime("%Y%m%d")}.xlsx',
                    mime="application/vnd.ms-excel"
                )

        except Exception as e:
            st.error(f" error {e}")
else:
    st.info(" check pls. 'ed.xlsx' file are uploaded ???")
