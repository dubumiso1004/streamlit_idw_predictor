import streamlit as st
import pandas as pd
import numpy as np

# -----------------------
# 위도/경도 변환 함수 (DMS → DD)
# -----------------------
def dms_to_dd(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

# -----------------------
# 데이터 불러오기 함수
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(dms_to_dd)
    df["Lon_dd"] = df["Lon"].apply(dms_to_dd)
    return df

# -----------------------
# IDW 보간 함수
# -----------------------
def idw_interpolation(df, lat, lon, variable, power=2):
    df = df.copy()
    df["distance"] = np.sqrt((df["Lat_dd"] - lat) ** 2 + (df["Lon_dd"] - lon) ** 2)
    df = df[df["distance"] != 0]  # 0 거리 제외 (자체 위치 방지)
    if df.empty:
        return None
    weights = 1 / (df["distance"] ** power)
    return np.sum(weights * df[variable]) / np.sum(weights)

# -----------------------
# UI 시작
# -----------------------
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템 (IDW 기반)")

df = load_data()

# 지도 중심 표시
df_map = df.rename(columns={"Lat_dd": "latitude", "Lon_dd": "longitude"})
st.map(df_map[["latitude", "longitude"]], zoom=17)

# 클릭 위치 입력 받기
clicked_lat = st.number_input("위도 (Lat_dd)", format="%.6f")
clicked_lon = st.number_input("경도 (Lon_dd)", format="%.6f")

if st.button("예측 실행"):
    svf = idw_interpolation(df, clicked_lat, clicked_lon, "SVF")
    gvi = idw_interpolation(df, clicked_lat, clicked_lon, "GVI")
    bvi = idw_interpolation(df, clicked_lat, clicked_lon, "BVI")
    pet = idw_interpolation(df, clicked_lat, clicked_lon, "PET")

    if None in (svf, gvi, bvi, pet):
        st.error("예측할 수 없습니다. 지도 내 측정 지점 근처를 선택해주세요.")
    else:
        st.success(f"☀️ 예측된 SVF: {svf:.3f}")
        st.success(f"🌿 예측된 GVI: {gvi:.3f}")
        st.success(f"🏢 예측된 BVI: {bvi:.3f}")
        st.success(f"🌡️ 예측된 PET: {pet:.2f}°C")
