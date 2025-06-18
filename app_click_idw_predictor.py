import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# 위경도 DMS → DD 변환 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# 거리 계산 함수 (Haversine)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

# IDW 예측 함수
def idw_predict(lat, lon, df, col, power=2):
    distances = df.apply(lambda row: haversine(lon, lat, row["Lon_dd"], row["Lat_dd"]), axis=1)
    if any(distances == 0):
        return df.loc[distances == 0, col].values[0]
    weights = 1 / distances**power
    return np.sum(weights * df[col]) / np.sum(weights)

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

# 📍 앱 실행
st.set_page_config(layout="wide")
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

df = load_data()

# 🗺️ 지도 시각화
df_map = df.rename(columns={"Lat_dd": "latitude", "Lon_dd": "longitude"})
st.map(df_map[["latitude", "longitude"]], zoom=17)

# 📌 지도 클릭
click = st.experimental_data_editor({"위도": [0.0], "경도": [0.0]}, num_rows="dynamic")
lat_click = st.number_input("위도 입력 (DD)", value=float(click["위도"][0]))
lon_click = st.number_input("경도 입력 (DD)", value=float(click["경도"][0]))

if st.button("예측 실행"):
    svf = idw_predict(lat_click, lon_click, df, "SVF")
    gvi = idw_predict(lat_click, lon_click, df, "GVI")
    bvi = idw_predict(lat_click, lon_click, df, "BVI")

    st.success(f"✅ 예측된 SVF: {svf:.3f}, GVI: {gvi:.3f}, BVI: {bvi:.3f}")
