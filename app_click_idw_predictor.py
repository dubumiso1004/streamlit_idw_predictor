
import streamlit as st
import pandas as pd
from idw_utils import idw_interpolation

@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250613.xlsx", sheet_name="gps 포함")
    df["Lat_dd"] = df["Lat"].apply(lambda x: float(str(x).replace(";", ".")))
    df["Lon_dd"] = df["Lon"].apply(lambda x: float(str(x).replace(";", ".")))
    return df

df = load_data()
center = [df["Lat_dd"].mean(), df["Lon_dd"].mean()]
st.title("🗺️ 지도 기반 보행자 열쾌적성 예측 시스템")
