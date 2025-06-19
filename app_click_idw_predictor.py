import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle  # ✅ cloudpickle 사용

# 모델 로드 함수
@st.cache_resource
def load_model():
    with open("pet_rf_model_full.pkl", "rb") as f:
        model = cloudpickle.load(f)
    return model

model = load_model()

# Streamlit UI
st.title("PET 예측 시스템")
st.write("SVF, GVI, BVI, 기온, 습도, 풍속을 기반으로 PET를 예측합니다.")

# 입력받기
svf = st.number_input("SVF (0~1)", min_value=0.0, max_value=1.0, value=0.3)
gvi = st.number_input("GVI (0~1)", min_value=0.0, max_value=1.0, value=0.4)
bvi = st.number_input("BVI (0~1)", min_value=0.0, max_value=1.0, value=0.2)
air_temp = st.number_input("기온 (°C)", value=25.0)
humidity = st.number_input("상대습도 (%)", value=50.0)
wind_speed = st.number_input("풍속 (m/s)", value=1.0)

# 예측 실행
if st.button("PET 예측하기"):
    input_data = pd.DataFrame([{
        "SVF": svf,
        "GVI": gvi,
        "BVI": bvi,
        "AirTemperature": air_temp,
        "Humidity": humidity,
        "WindSpeed": wind_speed
    }])
    prediction = model.predict(input_data)
    st.success(f"예측된 PET: {prediction[0]:.2f} °C")
