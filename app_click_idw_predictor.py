import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import math
import joblib

# -----------------------------
# 1. 좌표 변환 및 거리 계산 함수
# -----------------------------
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, str(dms_str).split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# -----------------------------
# 2. IDW 보간 함수
# -----------------------------
def idw_predict(df, lat, lon, var, power=2):
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    if any(df["dist"] == 0):
        return df.loc[df["dist"] == 0, var].values[0]
    df = df[df["dist"] > 0]
    df["weight"] = 1 / (df["dist"] ** power)
    return np.sum(df[var] * df["weight"]) / np.sum(df["weight"])

# -----------------------------
# 3. 데이터 및 모델 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["LON"].apply(dms_to_dd)
    return df.dropna(subset=["Lat_dd", "Lon_dd", "SVF", "GVI", "BVI"])

@st.cache_resource
def load_model():
    return joblib.load("pet_rf_model_gps.pkl")

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🧭 지도 클릭 기반 보행자 PET 예측 시스템")

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

click = st.experimental_data_editor(
    {"위도": [df["Lat_dd"].mean()], "경도": [df["Lon_dd"].mean()]},
    num_rows="dynamic"
)

lat = float(click["위도"][0])
lon = float(click["경도"][0])

if st.button("PET 예측 실행"):
    svf = idw_predict(df.copy(), lat, lon, "SVF")
    gvi = idw_predict(df.copy(), lat, lon, "GVI")
    bvi = idw_predict(df.copy(), lat, lon, "BVI")
    
    # 날씨 조건은 평균값 사용 또는 수동 지정 가능
    air_temp = df["AirTemperature"].mean()
    humidity = df["Humidity"].mean()
    wind_speed = df["WindSpeed"].mean()

    X_input = [[svf, gvi, bvi, air_temp, humidity, wind_speed]]
    pet = model.predict(X_input)[0]

    st.success(f"""
    ✅ 추정된 SVF: {svf:.3f}  
    ✅ 추정된 GVI: {gvi:.3f}  
    ✅ 추정된 BVI: {bvi:.3f}  
    🌡️ 예측된 PET: **{pet:.2f} °C**
    """)
