import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import st_folium

# 📌 DMS → DD 변환 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# 📌 Haversine 거리 함수 (위도 경도 거리)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# 📌 IDW 보간 함수
def idw_predict(df, lat, lon, var, power=2):
    df["dist"] = df.apply(lambda row: haversine(lat, lon, row["Lat_dd"], row["Lon_dd"]), axis=1)
    if any(df["dist"] == 0):
        return df.loc[df["dist"] == 0, var].values[0]
    df = df[df["dist"] > 0]
    df["weight"] = 1 / (df["dist"] ** power)
    return np.sum(df[var] * df["weight"]) / np.sum(df["weight"])

# 📌 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["LON"].apply(dms_to_dd)  # 열 이름 정확히 확인
    return df.dropna(subset=["Lat_dd", "Lon_dd", "SVF", "GVI", "BVI"])

# ===========================
# 🌡️ Streamlit App 시작
# ===========================
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

df = load_data()

# 📍 Folium 지도 생성
center = [df["Lat_dd"].mean(), df["Lon_dd"].mean()]
m = folium.Map(location=center, zoom_start=17)

# 측정지점 마커 추가
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Lat_dd"], row["Lon_dd"]],
        radius=4,
        color="red",
        fill=True,
        fill_opacity=0.7,
    ).add_to(m)

# 클릭 시 위경도 표시
m.add_child(folium.LatLngPopup())

# 📌 Streamlit에 지도 표시
st.markdown("### 지도에서 예측 지점을 클릭하세요")
map_data = st_folium(m, height=500, width=700)

# 📌 클릭한 지점 처리
if map_data and map_data["last_clicked"]:
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.info(f"선택된 위치: 위도 {lat:.6f}, 경도 {lon:.6f}")

    if st.button("예측 실행"):
        svf = idw_predict(df.copy(), lat, lon, "SVF")
        gvi = idw_predict(df.copy(), lat, lon, "GVI")
        bvi = idw_predict(df.copy(), lat, lon, "BVI")
        st.success(f"✅ 예측 결과\n- SVF: {svf:.3f}\n- GVI: {gvi:.3f}\n- BVI: {bvi:.3f}")
