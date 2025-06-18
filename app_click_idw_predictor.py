import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(layout="wide")

# 🌍 지도 시각화
import pydeck as pdk

# DMS → DD 변환 함수
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# 🔄 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

# 📌 IDW 보간 함수
def idw_predict(lat, lon, df, target_col, k=4):
    coords = df[["Lat_dd", "Lon_dd"]].values
    values = df[target_col].values

    distances = np.array([math.dist([lat, lon], pt) for pt in coords])
    nearest_idx = np.argsort(distances)[:k]

    nearest_dists = distances[nearest_idx]
    nearest_values = values[nearest_idx]

    if np.any(nearest_dists == 0):
        return nearest_values[nearest_dists == 0][0]

    weights = 1 / nearest_dists**2
    return np.sum(weights * nearest_values) / np.sum(weights)

# 📊 UI
df = load_data()
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

# 지도 중심 좌표
center = [df["Lat_dd"].mean(), df["Lon_dd"].mean()]
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=center[0],
            longitude=center[1],
            zoom=17,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[Lon_dd, Lat_dd]',
                get_color='[200, 30, 0, 160]',
                get_radius=8,
            ),
        ],
    )
)

# 입력 섹션
st.markdown("**예측할 위치의 위도/경도 (Decimal Degrees) 입력**")
lat_input = st.number_input("위도 (Latitude)", value=35.231743, format="%.6f")
lon_input = st.number_input("경도 (Longitude)", value=129.080665, format="%.6f")

if st.button("예측 실행"):
    svf = idw_predict(lat_input, lon_input, df, "SVF")
    gvi = idw_predict(lat_input, lon_input, df, "GVI")
    bvi = idw_predict(lat_input, lon_input, df, "BVI")

    st.success(f"✅ 예측된 SVF: {svf:.3f}, GVI: {gvi:.3f}, BVI: {bvi:.3f}")
