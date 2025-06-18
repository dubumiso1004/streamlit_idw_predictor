import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math
import joblib

# DMS(도;분;초) → Decimal Degrees 변환
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, str(dms_str).split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# Haversine 거리 계산 (위도 경도 거리)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# IDW 보간 함수
def idw_predict(df, lat, lon, var, power=2):
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    if any(df["dist"] == 0):
        return df.loc[df["dist"] == 0, var].values[0]
    df = df[df["dist"] > 0]
    df["weight"] = 1 / (df["dist"] ** power)
    return np.sum(df[var] * df["weight"]) / np.sum(df["weight"])

# 데이터 불러오기 (열 이름 수정)
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)  # 'LON' -> 'Lon'
    return df.dropna(subset=["Lat_dd", "Lon_dd", "SVF", "GVI", "BVI"])

# PET 예측 모델 로드
@st.cache_resource
def load_model():
    return joblib.load("pet_rf_model_full.pkl")

# 앱 UI 시작
st.title("🗺️ 지도 클릭 기반 보행자 열쾌적성 및 PET 예측 시스템")

df = load_data()
model = load_model()

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[Lon_dd, Lat_dd]',
    get_radius=25,
    get_color='[0, 128, 255]',
    pickable=True,
)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=df["Lat_dd"].mean(),
        longitude=df["Lon_dd"].mean(),
        zoom=17,
        pitch=0,
    ),
    layers=[layer],
))

# 수동 좌표 입력
lat = st.number_input("위도 (Latitude, DD)", value=df["Lat_dd"].mean(), format="%.6f")
lon = st.number_input("경도 (Longitude, DD)", value=df["Lon_dd"].mean(), format="%.6f")

if st.button("PET 예측 실행"):
    svf = idw_predict(df.copy(), lat, lon, "SVF")
    gvi = idw_predict(df.copy(), lat, lon, "GVI")
    bvi = idw_predict(df.copy(), lat, lon, "BVI")

    # 기상 변수는 평균값 사용 (원하면 입력 UI 추가 가능)
    air_temp = df["AirTemperature"].mean()
    humidity = df["Humidity"].mean()
    wind_speed = df["WindSpeed"].mean()

    X_input = [[svf, gvi, bvi, air_temp, humidity, wind_speed]]
    pet = model.predict(X_input)[0]

    st.success(f"""
    ✅ 예측된 SVF: {svf:.3f}  
    ✅ 예측된 GVI: {gvi:.3f}  
    ✅ 예측된 BVI: {bvi:.3f}  
    🌡️ 예측된 PET: **{pet:.2f} °C**
    """)
