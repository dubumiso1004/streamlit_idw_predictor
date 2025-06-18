import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math

# DMS(도;분;초)를 Decimal Degrees로 변환하는 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# Haversine 거리 계산 (위도, 경도 사이 거리 계산)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# IDW 보간 함수
def idw_predict(df, lat, lon, var, power=2):
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    
    if any(df["dist"] == 0):
        return df.loc[df["dist"] == 0, var].values[0]
    
    df = df[df["dist"] > 0]
    df["weight"] = 1 / (df["dist"] ** power)
    return np.sum(df[var] * df["weight"]) / np.sum(df["weight"])

# 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["LON"].apply(dms_to_dd)
    return df.dropna(subset=["Lat_dd", "Lon_dd", "SVF", "GVI", "BVI"])

# 📍 앱 실행
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

df = load_data()

# pydeck 지도 시각화
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[Lon_dd, Lat_dd]',
    get_radius=20,
    get_color='[255, 100, 100]',
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

# 클릭 기반 좌표 입력
click = st.experimental_data_editor(
    {"위도": [df["Lat_dd"].mean()], "경도": [df["Lon_dd"].mean()]},
    num_rows="dynamic"
)

lat = float(click["위도"][0])
lon = float(click["경도"][0])

# 예측 버튼
if st.button("예측 실행"):
    svf = idw_predict(df.copy(), lat, lon, "SVF")
    gvi = idw_predict(df.copy(), lat, lon, "GVI")
    bvi = idw_predict(df.copy(), lat, lon, "BVI")
    
    st.success(f"✅ 예측된 SVF: {svf:.3f}, GVI: {gvi:.3f}, BVI: {bvi:.3f}")
