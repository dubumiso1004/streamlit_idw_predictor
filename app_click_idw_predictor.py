import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
from idw_utils import idw_interpolation
import numpy as np

# --- 위도경도 문자열을 소수점 형식으로 변환 ---
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# --- 측정 데이터 불러오기 ---
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

# --- 보간 함수 ---
def interpolate_value(df, lat, lon, column, power=2):
    data = df[["Lat_dd", "Lon_dd", column]].dropna()
    lats = data["Lat_dd"].values
    lons = data["Lon_dd"].values
    values = data[column].values
    return idw_interpolation(lats, lons, values, lat, lon, power=power)

# --- 메인 앱 ---
st.set_page_config(layout="centered")
st.title("🌡️ 지도 클릭 기반 보행자 열쾌적성 예측")

st.markdown("""
🗺️ 지도 위를 클릭하면 해당 위치의 SVF, GVI, BVI 값을 IDW 방식으로 추정하고,

그 값을 기반으로 PET(Physiological Equivalent Temperature)를 예측합니다.
""")

# 1. 데이터 불러오기
df = load_data()

# 2. 지도 설정
m = folium.Map(location=[df["Lat_dd"].mean(), df["Lon_dd"].mean()], zoom_start=17)
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Lat_dd"], row["Lon_dd"]],
        radius=4,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6
    ).add_to(m)

# 3. 사용자 클릭 이벤트 처리
click_data = st_folium(m, width=700, height=500)

if click_data and click_data.get("last_clicked"):
    lat = click_data["last_clicked"]["lat"]
    lon = click_data["last_clicked"]["lng"]

    svf = interpolate_value(df, lat, lon, "SVF")
    gvi = interpolate_value(df, lat, lon, "GVI")
    bvi = interpolate_value(df, lat, lon, "BVI")

    # 간단한 선형 모델 예시 (원하는 예측 모델로 대체 가능)
    pet = 10 + svf * 15 - gvi * 8 + bvi * 6

    st.success(f"📍 클릭한 위치: {lat:.5f}, {lon:.5f}")
    st.markdown(f"- SVF(IDW 보간): `{svf:.3f}`")
    st.markdown(f"- GVI(IDW 보간): `{gvi:.3f}`")
    st.markdown(f"- BVI(IDW 보간): `{bvi:.3f}`")
    st.markdown(f"\n✅ 예측 PET: `{pet:.2f} °C`")
else:
    st.info("지도를 클릭하여 위치를 선택해주세요.")
