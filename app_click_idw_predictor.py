import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from idw_utils import idw_interpolation, dms_to_dd

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df.columns = df.columns.str.strip()  # 공백 제거

    # 위도/경도 변환
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

df = load_data()

# 중심 좌표 설정
center = [df["Lat_dd"].mean(), df["Lon_dd"].mean()]

# 📍 지도 출력
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

selected_point = st.map(df[["Lat_dd", "Lon_dd"]], zoom=17)

clicked_location = st.session_state.get("clicked_location", None)

# 지도 클릭 이벤트
def map_click_event(lat, lon):
    st.session_state["clicked_location"] = {"lat": lat, "lon": lon}

map_data = pd.DataFrame({
    "lat": df["Lat_dd"],
    "lon": df["Lon_dd"],
    "SVF": df["SVF"],
    "GVI": df["GVI"],
    "BVI": df["BVI"]
})

# 클릭 입력 받기
st.write("👉 지도 상 좌표를 클릭해 보간 결과를 확인하세요.")
clicked = st.map(map_data, zoom=17)

# 실제 좌표 클릭 처리
if clicked_location:
    click_lat = clicked_location["lat"]
    click_lon = clicked_location["lon"]

    svf = idw_interpolation(df, "Lat_dd", "Lon_dd", "SVF", click_lat, click_lon)
    gvi = idw_interpolation(df, "Lat_dd", "Lon_dd", "GVI", click_lat, click_lon)
    bvi = idw_interpolation(df, "Lat_dd", "Lon_dd", "BVI", click_lat, click_lon)

    st.success(f"📍 선택 위치: 위도 {click_lat:.6f}, 경도 {click_lon:.6f}")
    st.write(f"🌤️ 추정 SVF: `{svf:.3f}`")
    st.write(f"🌿 추정 GVI: `{gvi:.3f}`")
    st.write(f"🏢 추정 BVI: `{bvi:.3f}`")
