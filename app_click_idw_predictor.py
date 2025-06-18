import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium

# 좌표 문자열 DMS → DD 변환 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# 거리 계산 (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c * 1000  # meter

# IDW 보간
def idw_predict(df, lat, lon, var, power=2):
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    df = df[df["dist"] > 0]
    df["weight"] = 1 / (df["dist"] ** power)
    return np.sum(df[var] * df["weight"]) / np.sum(df["weight"])

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

# 앱 시작
st.set_page_config(layout="wide")
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

df = load_data()

# 지도 생성
m = folium.Map(location=[df["Lat_dd"].mean(), df["Lon_dd"].mean()], zoom_start=17)
folium.LatLngPopup().add_to(m)

# 지도 표시
st_data = st_folium(m, width=700, height=500)

# 좌표 클릭 시 예측 실행
if st_data["last_clicked"] is not None:
    lat = st_data["last_clicked"]["lat"]
    lon = st_data["last_clicked"]["lng"]
    
    svf = idw_predict(df, lat, lon, "SVF")
    gvi = idw_predict(df, lat, lon, "GVI")
    bvi = idw_predict(df, lat, lon, "BVI")

    st.success(f"📍 선택된 위치: 위도 {lat:.6f}, 경도 {lon:.6f}")
    st.markdown(f"✅ 예측된 SVF: `{svf:.3f}`, GVI: `{gvi:.3f}`, BVI: `{bvi:.3f}`")
else:
    st.info("지도를 클릭하여 예측할 위치를 선택하세요.")
